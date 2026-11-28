"""Offline ranking test for search scorers: how well does a scorer pick among candidate turns?

    .venv/bin/python experiments/herox_rank.py build --roots 200 --out runs/herox/rank-gt.npz
    .venv/bin/python experiments/herox_rank.py score --data runs/herox/rank-gt.npz \
        runs/herox/stageA.npz runs/herox/scorer-u2.npz runs/herox/scorer-u4b.npz

Search (rl/search.py, plain d=1) plays K candidate turns from a root, one sampled opponent reply each, and keeps the
turn its scorer rates best at the searcher's next decision. This test rebuilds that situation once, with ground
truth, so any scorer can be measured in minutes instead of hours of games.

build:
  1. Roots: start-of-turn positions from Plato self-play (temperature 1), at most --per-game per game.
  2. Candidates: K turns per root from Plato, lane 0 greedy and lanes 1.. sampled at temperature 1 (as search).
  3. Rollouts: M continuations per candidate, Plato at temperature 1 for both sides, for --rounds of the searcher's
     turns. Rollout 0 is what search sees: the searcher's own view at its next decision (after one sampled reply);
     each scorer scores it. Rollouts 1..M-1 give the ground truth: the outcome if the game ended (0 at the round
     cap), else Plato's privileged critic (both views, never trained here) at the horizon, from the searcher's side.
     The scored rollout is not part of the ground truth, so a scorer is not rewarded for fitting its own sample.
  A candidate whose turn, or rollout 0's reply, ends the game is scored by that outcome, as search does.

score: per scorer and root, the candidate it picks (ties keep the greedy lane 0, as search) and
  gain = truth(picked) - truth(greedy), the expected improvement per searched turn in critic units;
  also how often it overrides greedy, and its pairwise ordering accuracy on candidate pairs whose ground truths differ
  by more than --margin (a tie counts half: in the opening many of Plato's sampled turns equal its greedy one). Baselines: always greedy (0), a uniformly random candidate, and the best candidate in
  hindsight (an upper bound, biased up by the ground truth's own noise). Scorers are compared root by root (paired
  bootstrap over roots).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SIZE = 14
PLATO = ROOT / 'weights' / 'plato' / 'final.npz'


def _plato():
    from polytopia_bot.rl.distributed import _critic_template, _template_params, network_spec
    from polytopia_bot.rl.runner import load_checkpoint
    spec = network_spec(PLATO)
    return (load_checkpoint(PLATO, _template_params(**spec)),
            load_checkpoint(PLATO.with_name('final-critic.npz'), _critic_template(**spec)))


def roots_from_selfplay(policy, count, per_game, seed, seed_base, lanes=32, max_round=45):
    """Start-of-turn states (single-game, gpu/state.py layout) from Plato self-play at temperature 1."""
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.search import first
    rng = np.random.default_rng(seed)
    engine = RustGames(SIZE, lanes)
    act = jax.jit(jax.vmap(lambda f, m, k: network.sample(policy, f, m, k, size=SIZE)[0]))
    seeds = iter(range(seed_base, seed_base + 10_000_000))
    states, prepared = engine.reset(np.array([next(seeds) for _ in range(lanes)]))
    key = jax.random.key(seed)
    # each game draws per_game target rounds uniformly in [1, max_round]; the first turn of each target round (either
    # seat) becomes a root, so roots spread over the whole game rather than its first turns
    draw = lambda: set(rng.choice(np.arange(1, max_round + 1), per_game, replace=False).tolist())
    last, targets, roots = [None] * lanes, [draw() for _ in range(lanes)], []
    while len(roots) < count:
        mover, turn = np.asarray(states.current_player), np.asarray(states.turn)
        full = None
        for i in range(lanes):
            now = (int(turn[i]), int(mover[i]))
            if last[i] != now:
                last[i] = now
                if now[0] in targets[i]:
                    targets[i].discard(now[0])
                    full = engine.states() if full is None else full
                    roots.append(first(full, i))
        key, k = jax.random.split(key)
        choice = act(prepared.features, prepared.masks, jax.random.split(k, lanes))
        tick, status = engine.tick(states, choice, np.array([next(seeds) for _ in range(lanes)]))
        for i in np.nonzero(np.asarray(tick.done))[0]:
            last[i], targets[i] = None, draw()
        states, prepared = tick.states, tick.prepared
    return roots[:count]


class Runner:
    """Lockstep engine-rs lanes sampled by Plato (rl/search.py's Lanes and jitted sampler)."""

    def __init__(self, policy, critic, lanes):
        import jax
        from polytopia_bot.rl import transformer_v3
        from polytopia_bot.rl.cpu_engine import view_features, view_layout
        from polytopia_bot.rl.observation import Features
        from polytopia_bot.rl.search import Lanes, _Nets
        self.policy, self.lanes = policy, Lanes(SIZE, lanes)
        self.nets = _Nets(SIZE)
        layout = view_layout(SIZE)

        def critic_value(own, opp):
            a, b = view_features(own, layout, SIZE), view_features(opp, layout, SIZE)
            import jax.numpy as jnp
            f = Features(jnp.concatenate((a.board, b.board), -1), jnp.concatenate((a.global_state, b.global_state), -1))
            return transformer_v3.value(policy, critic, f)
        self.critic = jax.jit(jax.vmap(critic_value))

    def play(self, start, me, keys, greedy, *, stop_changes, record_at=None):
        """Plays every lane from `start` (batched states) until its mover has changed `stop_changes` times or the game
        ends. me [k]: the searcher's seat per lane. Returns dict(states, ended, outcome, changes, view, critic):
        view = the mover's packed view when the change count first reached `record_at` (searcher's next decision),
        critic = Plato's critic at the stop, from me's side (0 where the game ended)."""
        from polytopia_bot.rl.search import NOOP
        lanes, k = self.lanes, len(me)
        lanes.load(start)
        changes, done = np.zeros(k, int), np.zeros(k, bool)
        ended, outcome = np.zeros(k, bool), np.zeros(k, np.float32)
        view, seen = None, np.zeros(k, bool)
        zeros = np.zeros(k, np.int32)
        temp, force = np.ones(k, np.float32), np.zeros(k, bool)
        while True:
            if record_at is not None and view is None:
                view = np.zeros((k,) + np.shape(lanes.views()[0]), np.int16)
            if record_at is not None:
                hit = ~done & (changes == record_at) & ~seen
                if hit.any():
                    v = lanes.views()
                    view[hit], seen[hit] = v[hit], True
            done |= changes >= stop_changes
            if done.all():
                break
            mover = lanes.scalars()['current_player']
            pair = self.nets.split(keys)
            keys, draws = pair[:, 0], pair[:, 1]
            choice, _ = self.nets.sample(self.policy, lanes.views(), lanes.masks(), draws, temp, greedy & (changes == 0),
                                         force, zeros, zeros)
            fam, src, tgt = (np.asarray(x) for x in choice)
            go = ~done
            lanes.step(np.where(go, fam, NOOP[0]), np.where(go, src, NOOP[1]), np.where(go, tgt, NOOP[2]))
            after = lanes.scalars()
            fin = go & (after['terminated'] | after['truncated'])
            if fin.any():
                r = lanes.rewards()
                outcome = np.where(fin & after['terminated'], r[np.arange(k), me - 1], outcome).astype(np.float32)
                ended |= fin
                done |= fin
            changes += go & ~fin & (after['current_player'] != mover)
        critic = np.zeros(k, np.float32)
        live = ~ended
        if live.any():
            # at the stop it is me's decision (odd change counts after the candidate) or the opponent's (the
            # candidate turn itself); the critic is evaluated from me's side either way
            one, two = lanes.views(1), lanes.views(2)
            own = np.where((me == 1)[:, None], one, two)
            opp = np.where((me == 1)[:, None], two, one)
            mine = np.asarray(self.critic(own, opp))
            critic = np.where(live, mine, 0.).astype(np.float32)
        return dict(states=lanes.states(), ended=ended, outcome=outcome, changes=changes, view=view, seen=seen,
                    critic=critic)


def build(args):
    import jax
    from polytopia_bot.rl.search import take
    policy, critic = _plato()
    started = time.time()
    roots = roots_from_selfplay(policy, args.roots, args.per_game, args.seed, args.seed_base)
    print(f'{len(roots)} roots ({time.time() - started:.0f}s); rounds {sorted(int(r.turn) for r in roots)[::20]}', flush=True)
    R, K, M = len(roots), args.candidates, args.rollouts
    stacked = jax.tree.map(lambda *a: np.stack([np.asarray(x) for x in a]), *roots)
    me_root = np.array([int(r.current_player) for r in roots])
    key = jax.random.key(args.seed + 1)
    chunk = args.chunk

    # candidates: lane 0 of each root greedy, the rest sampled; play until the mover changes once
    cand_states, cand_ended, cand_outcome = [], [], []
    cand_index = np.repeat(np.arange(R), K)
    runner = Runner(policy, critic, chunk)
    for c in range(0, R * K, chunk):
        idx = cand_index[c:c + chunk]
        n = len(idx)
        pad = np.concatenate((idx, np.full(chunk - n, idx[-1])))
        key, k = jax.random.split(key)
        greedy = (np.arange(c, c + chunk) % K) == 0
        out = runner.play(take(stacked, pad), me_root[pad], jax.random.split(k, chunk), greedy, stop_changes=1)
        cand_states.append(take(out['states'], np.arange(n)))
        cand_ended.append(out['ended'][:n])
        cand_outcome.append(out['outcome'][:n])
    cand = jax.tree.map(lambda *a: np.concatenate(a), *cand_states)
    cand_ended, cand_outcome = np.concatenate(cand_ended), np.concatenate(cand_outcome)
    print(f'candidates: {R * K} turns, {cand_ended.sum()} ended the game ({time.time() - started:.0f}s)', flush=True)

    # rollouts: M per candidate; stop at the searcher's decision args.rounds turns later (2 * rounds - 1 changes)
    stop = 2 * args.rounds - 1
    n_roll = R * K * M
    roll_cand = np.repeat(np.arange(R * K), M)
    value = np.zeros(n_roll, np.float32)
    ended, outcome = np.zeros(n_roll, bool), np.zeros(n_roll, np.float32)
    views, reached_all, rollouts_started = None, np.zeros(n_roll, bool), time.time()
    for c in range(0, n_roll, chunk):
        idx = roll_cand[c:c + chunk]
        n = len(idx)
        pad = np.concatenate((idx, np.full(chunk - n, idx[-1])))
        key, k = jax.random.split(key)
        out = runner.play(take(cand, pad), me_root[cand_index[pad]], jax.random.split(k, chunk), np.zeros(chunk, bool),
                          stop_changes=stop, record_at=1)
        if views is None:
            views = np.zeros((n_roll,) + out['view'].shape[1:], np.int16)
        views[c:c + n] = out['view'][:n]
        value[c:c + n], ended[c:c + n], outcome[c:c + n] = out['critic'][:n], out['ended'][:n], out['outcome'][:n]
        # a reply that ended the game before the searcher's next decision leaves no view (scored by the outcome)
        reached_all[c:c + n] = out['seen'][:n]
        if c // chunk % 5 == 0:
            done, spent = c + n, time.time() - rollouts_started
            print(f'rollouts {done}/{n_roll} ({time.time() - started:.0f}s, ~{spent / done * (n_roll - done) / 60:.0f} '
                  f'min left)', flush=True)
    # candidate-level terminal: the candidate turn itself ended the game
    term = cand_ended[roll_cand]
    result = np.where(term, cand_outcome[roll_cand], np.where(ended, outcome, value))
    result = result.reshape(R, K, M)
    np.savez_compressed(args.out, views=views.reshape(R, K, M, -1), reached=reached_all.reshape(R, K, M),
                        result=result, candidate_ended=cand_ended.reshape(R, K),
                        candidate_outcome=cand_outcome.reshape(R, K), root_turn=np.array([int(r.turn) for r in roots]),
                        root_seat=me_root, rounds=args.rounds)
    truth = result[:, :, 1:].mean(-1)
    print(json.dumps(dict(roots=R, candidates=K, rollouts=M, rounds=args.rounds, seconds=round(time.time() - started),
                          truth_spread_within_root=round(float(truth.std(1).mean()), 4),
                          truth_noise_per_candidate=round(float(result[:, :, 1:].std(-1).mean() / np.sqrt(M - 1)), 4),
                          out=args.out), indent=1))


def _scores(views, reached, terminal_value, scorer):
    """Scores of rollout-0 views [n] by a scorer callable (packed views -> values), terminal where not reached."""
    out = np.array(terminal_value, np.float32)
    live = np.nonzero(reached)[0]
    for i in range(0, len(live), 256):
        b = live[i:i + 256]
        out[b] = np.asarray(scorer(views[b]))
    return out


def score(args):
    import jax
    from polytopia_bot.rl import herox, search
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    d = np.load(args.data)
    result, reached = d['result'], d['reached']
    R, K, M = result.shape
    truth = result[:, :, 1:].mean(-1)                        # [R, K]
    views0 = d['views'][:, :, 0].reshape(R * K, -1)
    reached0 = reached[:, :, 0].reshape(-1) & ~d['candidate_ended'].reshape(-1)
    terminal0 = result[:, :, 0].reshape(-1)                  # outcome where rollout 0 ended (or the candidate did)
    layout = view_layout(SIZE)
    picks = {}

    def pick(s):
        s = s.reshape(R, K)
        best = s.argmax(1)
        return np.where(s[np.arange(R), best] > s[:, 0], best, 0)    # ties keep greedy

    for path in args.scorers:
        params, _ = herox.load(path)
        f = jax.jit(jax.vmap(lambda v: search._value(params, view_features(v, layout, SIZE))))
        s = _scores(views0, reached0, terminal0, f)
        picks[Path(path).stem] = (pick(s), s.reshape(R, K))
        print(f'scored {path}', flush=True)
    rng = np.random.default_rng(0)
    picks['random'] = (rng.integers(0, K, R), None)
    picks['hindsight (upper bound)'] = (truth.argmax(1), truth)
    base = truth[:, 0]
    idx = rng.integers(0, R, (4000, R))
    rows = {}
    for name, (p, s) in picks.items():
        gain = truth[np.arange(R), p] - base
        boot = gain[idx].mean(1)
        row = dict(gain=round(float(gain.mean()), 4), ci95=[round(float(np.percentile(boot, 2.5)), 4),
                                                             round(float(np.percentile(boot, 97.5)), 4)],
                   override=round(float((p != 0).mean()), 3))
        if s is not None:
            i, j = np.triu_indices(K, 1)
            dt = truth[:, i] - truth[:, j]
            ds = s[:, i] - s[:, j]
            big = np.abs(dt) > args.margin
            right = (np.sign(ds) == np.sign(dt)) + .5 * (ds == 0)       # a tie (identical candidates) counts half
            row['pair_accuracy'] = round(float((right * big).sum() / max(big.sum(), 1)), 4)
            row['pairs'] = int(big.sum())
        rows[name] = row
        print(f'{name:28s} ' + json.dumps(row), flush=True)
    names = [Path(p).stem for p in args.scorers]
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ga = truth[np.arange(R), picks[names[a]][0]] - truth[np.arange(R), picks[names[b]][0]]
            boot = ga[idx].mean(1)
            print(f'{names[a]} - {names[b]}: {ga.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, '
                  f'{np.percentile(boot, 97.5):+.4f}]  same pick {np.mean(picks[names[a]][0] == picks[names[b]][0]):.0%}')
    if args.out:
        Path(args.out).write_text(json.dumps(dict(data=args.data, roots=R, candidates=K, rollouts=M, rows=rows), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--roots', type=int, default=200)
    b.add_argument('--per-game', type=int, default=4)
    b.add_argument('--candidates', type=int, default=12)
    b.add_argument('--rollouts', type=int, default=6)
    b.add_argument('--rounds', type=int, default=3, help="searcher's turns until the ground-truth horizon")
    b.add_argument('--chunk', type=int, default=256, help='engine lanes stepped together')
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--seed-base', type=int, default=7_000_000)
    b.add_argument('--out', default='runs/herox/rank-gt.npz')
    b.set_defaults(run=build)
    s = sub.add_parser('score')
    s.add_argument('--data', default='runs/herox/rank-gt.npz')
    s.add_argument('--margin', type=float, default=.05, help='pairs whose ground truths differ by more than this')
    s.add_argument('--out', default='')
    s.add_argument('scorers', nargs='+')
    s.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
