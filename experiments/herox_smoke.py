"""heroX smoke test on CPU: self-play data collection, then frozen-trunk training of the new heads.

    .venv/bin/python experiments/herox_smoke.py collect --games 16 --lanes 8 --out runs/herox/smoke-data.npz
    .venv/bin/python experiments/herox_smoke.py train --data runs/herox/smoke-data.npz --steps 200

collect: heroX plays itself on engine-rs (sampled moves; a plan chosen by the macro head at the start of each turn).
Every decision is stored as the mover's and the opponent's packed views, the action path and the plan; when the game
ends each row is labelled with the outcome (a capped game is a draw), the final score margin, the turns remaining and
the final tile owners. Legal masks come from the true state, as in Plato's training (docs/plato-t3.md caveat).

train: Adam on the new heads and the WDL critic only (Plato's weights stay frozen), herox.head_loss, games split
80/20 into train and validation. Reports validation losses before and after, and the outcome-frequency baseline.

This checks the pipeline end to end. It is far too small to train anything useful.
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


def _model(init):
    from polytopia_bot.rl import herox
    if init:
        return herox.load(init)
    import jax
    from polytopia_bot.rl.distributed import _critic_template, _template_params, network_spec
    from polytopia_bot.rl.runner import load_checkpoint
    plato = ROOT / 'weights' / 'plato' / 'final.npz'
    spec = network_spec(plato)
    policy = load_checkpoint(plato, _template_params(**spec))
    critic = load_checkpoint(plato.with_name('final-critic.npz'), _critic_template(**spec))
    return herox.from_plato(policy, critic, jax.random.key(0))


def collect(args):
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import herox
    from polytopia_bot.rl.actions import FAMILIES
    from polytopia_bot.rl.rust_engine import RustGames
    params, _ = _model(args.init)
    MOVE = FAMILIES.index('move')
    lanes = args.lanes
    engine = RustGames(SIZE, lanes, both_views=True, max_turns=args.max_turns)
    act = jax.jit(jax.vmap(lambda f, m, k, z: herox.act(params, f, m, k, z, size=SIZE)[:3]))
    choose = jax.jit(jax.vmap(lambda f, z, k: herox.choose_plan(params, f, z, k)[0]))
    rng = np.random.default_rng(args.seed)
    next_seed = iter(range(args.seed_base, args.seed_base + 10_000_000))
    states, prepared = engine.reset(np.array([next(next_seed) for _ in range(lanes)]))
    key = jax.random.key(args.seed)
    plans = np.zeros((lanes, 2), np.int32)
    last = [None] * lanes                       # (turn, mover) of the lane's previous decision
    buffers = [[] for _ in range(lanes)]
    decisions = np.zeros(lanes, np.int64)            # per lane: decisions and last round of the current game
    rounds = np.zeros(lanes, np.int64)
    rows, finished, ticks, shards = [], [], 0, []
    # per lane: (seat, turn) -> [action counts per family..., summed move advance, moves], over EVERY decision
    turn_stats = [{} for _ in range(lanes)]
    turns = []
    families = len(FAMILIES)
    cells = np.arange(SIZE * SIZE)
    row_of, col_of = cells // SIZE, cells % SIZE
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        # one compressed shard per --shard-games finished games, so memory stays bounded
        if rows:
            path = out.with_name(f'{out.stem}-{len(shards):03d}.npz')
            extra = {f'turn_{k}': np.stack([t[k] for t in turns]) for k in turns[0]} if turns else {}
            np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in rows[0]}, **extra)
            turns.clear()
            shards.append(dict(path=path.name, rows=len(rows)))
            rows.clear()
    counts = np.zeros(3, np.int64)
    started = time.time()
    while len(finished) < args.games:
        mover = np.asarray(states.current_player)          # 1 or 2
        turn = np.asarray(states.turn)
        new_turn = np.array([last[i] != (int(turn[i]), int(mover[i])) for i in range(lanes)])
        key, k_plan, k_act = jax.random.split(key, 3)
        if new_turn.any():
            previous = plans[np.arange(lanes), mover - 1]
            chosen = np.asarray(choose(prepared.features, jnp.asarray(previous), jax.random.split(k_plan, lanes)))
            plans[np.arange(lanes), mover - 1] = np.where(new_turn, chosen, previous)
        plan = plans[np.arange(lanes), mover - 1]
        choice, path, _ = act(prepared.features, prepared.masks, jax.random.split(k_act, lanes), jnp.asarray(plan))
        choice = tuple(np.asarray(c) for c in choice)
        path = tuple(np.asarray(p) for p in path)
        kept = (rng.random(lanes) < args.keep) | new_turn       # the first decision of every turn is always kept
        owners = np.asarray(states.tiles.owner)
        # opponent-state labels from the true state (heroX belief heads), and the searcher's own for the baselines
        tech, stars = np.asarray(states.players.tech), np.asarray(states.players.stars)
        income = np.stack([np.asarray(engine.batch.income(1)), np.asarray(engine.batch.income(2))], 1)
        city_owner, city_level = np.asarray(states.cities.owner), np.asarray(states.cities.level)
        unit_owner = np.asarray(states.units.owner)
        for i in range(lanes):
            stats = turn_stats[i].setdefault((int(mover[i]), int(turn[i])), np.zeros(families + 2))
            stats[choice[0][i]] += 1
            if choice[0][i] == MOVE:
                # advance: Chebyshev distance to the centroid of the opponent's territory, before minus after
                enemy = owners[i] == 3 - int(mover[i])
                if enemy.any():
                    cy, cx = row_of[enemy].mean(), col_of[enemy].mean()
                    dist = lambda t: max(abs(row_of[t] - cy), abs(col_of[t] - cx))
                    stats[families] += dist(choice[1][i]) - dist(choice[2][i])
                    stats[families + 1] += 1
            last[i] = (int(turn[i]), int(mover[i]))
            decisions[i] += 1
            rounds[i] = max(rounds[i], int(turn[i]))
            if not kept[i]:
                continue
            buffers[i].append(dict(view=np.asarray(prepared.view[i]), opp_view=np.asarray(prepared.opp_view[i]),
                                   family=choice[0][i], source=choice[1][i], target=choice[2][i],
                                   family_mask=path[0][i], source_mask=path[1][i], target_mask=path[2][i],
                                   plan=plan[i], seat=int(mover[i]), turn=int(turn[i]), first=bool(new_turn[i]),
                                   **_opponent_state(i, int(mover[i]), tech, stars, income, city_owner, city_level,
                                                     unit_owner)))
        scores = np.asarray(states.players.score)          # the last pre-action state (the tick resets done lanes)
        tick, status = engine.tick(states, choice, np.array([next(next_seed) for _ in range(lanes)]))
        if np.any(np.asarray(status) != 0):
            raise RuntimeError('illegal action in self-play')
        ticks += 1
        for i in np.nonzero(np.asarray(tick.done))[0]:
            rewards, capped = np.asarray(tick.rewards)[i], bool(np.asarray(tick.truncated)[i])
            game = len(finished)
            final_turn = int(rounds[i])
            for r in buffers[i]:
                me, other = r['seat'], 3 - r['seat']
                outcome = int(herox.outcome_class(rewards[me - 1], capped))
                counts[outcome] += 1
                rows.append(dict(r, game=game, outcome=outcome,
                                 margin=float(scores[i, me - 1] - scores[i, other - 1]),
                                 turns_left=final_turn - r['turn'],
                                 ownership=np.where(owners[i] == me, 1, np.where(owners[i] == other, -1, 0)
                                                    ).astype(np.int8)))
            for (seat, t), stats in sorted(turn_stats[i].items()):
                turns.append(dict(game=game, seat=seat, turn=t, counts=stats[:families].astype(np.int16),
                                  advance=np.float32(stats[families]), advance_moves=np.int16(stats[families + 1])))
            turn_stats[i] = {}
            finished.append(dict(game=game, decisions=int(decisions[i]), kept=len(buffers[i]), rounds=final_turn,
                                 capped=capped, winner=0 if capped else int(np.argmax(rewards)) + 1))
            buffers[i], last[i], decisions[i], rounds[i] = [], None, 0, 0
            plans[i] = 0
            print(f'game {game + 1}/{args.games}: {finished[-1]}  '
                  f'({time.time() - started:.0f}s, {ticks} ticks)', flush=True)
            if len(finished) % args.shard_games == 0:
                flush()
            if len(finished) >= args.games:
                break
        states, prepared = tick.states, tick.prepared
    flush()
    summary = dict(rows=int(counts.sum()), keep=args.keep, shards=shards, games=finished,
                   seconds=round(time.time() - started, 1),
                   outcomes={n: int(counts[c]) for n, c in
                             (('win', herox.WIN), ('draw', herox.DRAW), ('loss', herox.LOSS))})
    out.with_suffix('.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'games'}, indent=1))


def _opponent_state(i, me, tech, stars, income, city_owner, city_level, unit_owner):
    """Row labels for heroX's belief heads: the opponent's tech, stars, income, cities, total city level and units at
    this decision (true state), and the mover's own tech, stars and income (for the rule-based baselines)."""
    o = 3 - me
    theirs = city_owner[i] == o
    return dict(opp_tech=tech[i, o - 1].astype(bool), opp_stars=np.int16(stars[i, o - 1]),
                opp_income=np.int16(income[i, o - 1]), opp_cities=np.int16(theirs.sum()),
                opp_city_levels=np.int16(city_level[i][theirs].sum()), opp_units=np.int16((unit_owner[i] == o).sum()),
                me_tech=tech[i, me - 1].astype(bool), me_stars=np.int16(stars[i, me - 1]),
                me_income=np.int16(income[i, me - 1]))


def _belief_baselines(data, rows, train_rows):
    """Accuracy of belief.py's rules (enemy tech = own tech; enemy stars = own stars - own income) and of the most
    frequent training bucket, on the same targets as herox.head_loss's tech_accuracy / <economy>_accuracy."""
    from polytopia_bot.rl import herox
    valid = herox.tech_mask()
    out = dict(rows=int(len(rows)), tech_copy_own=float((data['me_tech'][rows] == data['opp_tech'][rows])[:, valid].mean()),
               tech_majority=float(np.maximum(data['opp_tech'][train_rows][:, valid].mean(0),
                                              1 - data['opp_tech'][train_rows][:, valid].mean(0)).mean()))
    bucket = lambda k, v: np.clip(np.asarray(v) // herox.ECONOMY_SCALE[k], 0, herox.ECONOMY_BUCKETS - 1)
    truth = {k: bucket(k, data[f'opp_{k}'][rows]) for k in herox.ECONOMY}
    rule = bucket('stars', np.maximum(data['me_stars'][rows].astype(int) - data['me_income'][rows], 0))
    out['stars_own_minus_income'] = float((rule == truth['stars']).mean())
    out['income_copy_own'] = float((bucket('income', data['me_income'][rows]) == truth['income']).mean())
    for k in herox.ECONOMY:
        common = np.bincount(bucket(k, data[f'opp_{k}'][train_rows]), minlength=herox.ECONOMY_BUCKETS).argmax()
        out[f'{k}_majority'] = float((truth[k] == common).mean())
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def load_data(path):
    """A single .npz, or every shard <stem>-NNN.npz that collect wrote for that path."""
    path = Path(path)
    files = sorted(path.parent.glob(f'{path.stem}-[0-9][0-9][0-9].npz')) or [path]
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def train(args):
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import herox, ppo
    from polytopia_bot.rl.actions import Choice, PathMasks
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    from polytopia_bot.rl.observation import Features
    params, critic = _model(args.init)
    # --data a,b,...: datasets concatenated, game ids offset per dataset; each dataset is split into train and
    # validation games on its own (so one dataset's split is the same as when it is trained alone)
    parts = [load_data(d) for d in args.data.split(',')]
    belief_keys = ('opp_tech', 'opp_stars', 'opp_income', 'opp_cities', 'opp_city_levels', 'opp_units', 'me_tech',
                   'me_stars', 'me_income')
    if any('opp_tech' in p for p in parts):
        # datasets collected before the belief labels: placeholders, masked out of the belief losses by has_belief
        for p in parts:
            rows = len(p['game'])
            p['has_belief'] = np.full(rows, 'opp_tech' in p)
            for k in belief_keys:
                if k not in p:
                    p[k] = np.zeros((rows, herox.TECH_SLOTS), bool) if k.endswith('tech') else np.zeros(rows, np.int16)
    keys = sorted(set.intersection(*(set(p) for p in parts)) - {k for k in parts[0] if k.startswith('turn_')})
    rng = np.random.default_rng(args.seed)
    val_parts = []
    for i, part in enumerate(parts):
        games = np.unique(part['game'])
        held = np.random.default_rng(args.seed).choice(games, max(1, len(games) // 5), replace=False)
        val_parts.append(np.isin(part['game'], held))
        part['game'] = part['game'] + 1_000_000 * i
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    is_val = np.concatenate(val_parts)
    del parts
    layout = view_layout(SIZE, 2)
    encode = jax.vmap(lambda v: view_features(v, layout, SIZE))

    def batch_of(index):
        own, opp = encode(jnp.asarray(data['view'][index])), encode(jnp.asarray(data['opp_view'][index]))
        return dict(features=Features(jnp.concatenate((own.board, opp.board), -1),
                                      jnp.concatenate((own.global_state, opp.global_state), -1)),
                    choice=Choice(*(jnp.asarray(data[k][index]) for k in ('family', 'source', 'target'))),
                    masks=PathMasks(*(jnp.asarray(data[k][index]) for k in ('family_mask', 'source_mask', 'target_mask'))),
                    plan=jnp.asarray(data['plan'][index]), outcome=jnp.asarray(data['outcome'][index]),
                    score=herox.score_bucket(jnp.asarray(data['margin'][index])),
                    turns=herox.turns_bucket(jnp.asarray(data['turns_left'][index])),
                    ownership=jnp.asarray(data['ownership'][index], jnp.float32),
                    plan_label=jnp.full(len(index), -1, jnp.int32),   # no plan labels yet
                    **belief_of(index))

    def belief_of(index):
        if 'opp_tech' not in data:
            return {}
        return dict(opp_tech=jnp.asarray(data['opp_tech'][index]), has_belief=jnp.asarray(data['has_belief'][index]),
                    economy=herox.economy_buckets(*(jnp.asarray(data[f'opp_{k}'][index], jnp.int32)
                                                    for k in herox.ECONOMY)))

    games = np.unique(data['game'])
    train_rows, val_rows = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    val_rows = rng.choice(val_rows, min(len(val_rows), args.val_rows), replace=False)
    print(f'{len(data["game"])} rows from {len(games)} games: {len(train_rows)} train, {len(val_rows)} validation '
          f'({len(np.unique(data["game"][is_val]))} held-out games)')

    names = herox.VALUE_HEADS if args.stage == 'value' else tuple(params['heads'])
    train_heads, frozen_heads = herox.head_group(params['heads'], names)
    print(f'stage {args.stage}: training heads {sorted(train_heads)} and the critic')
    n = args.unfreeze_blocks
    blocks = params['policy']['transformer']
    if n:
        # A SCORING copy: the last n transformer blocks and the final norm train with the value losses, so this
        # checkpoint's policy is no longer Plato's. Use it as search's value_params, never to choose moves.
        print(f'unfreezing the last {n} of {len(blocks)} transformer blocks and the final norm (scoring copy)')
    tail = dict(blocks=tuple(blocks[len(blocks) - n:]), norm=params['policy']['v3_final_norm']) if n else {}

    def merged(heads, tail):
        policy = params['policy']
        if tail:
            policy = dict(policy, transformer=tuple(blocks[:len(blocks) - n]) + tuple(tail['blocks']),
                          v3_final_norm=tail['norm'])
        return dict(params, policy=policy, heads={**frozen_heads, **heads})

    def loss(trainable, batch):
        heads, wdl_critic, tail = trainable
        return herox.head_loss(merged(heads, tail), wdl_critic, batch)

    grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    metrics_of = jax.jit(lambda t, b: loss(t, b)[1])

    def validate(trainable):
        parts = [metrics_of(trainable, batch_of(val_rows[i:i + args.batch])) for i in range(0, len(val_rows), args.batch)]
        sizes = [min(args.batch, len(val_rows) - i) for i in range(0, len(val_rows), args.batch)]
        return {k: float(sum(float(p[k]) * n for p, n in zip(parts, sizes)) / sum(sizes)) for k in parts[0]}

    trainable = (train_heads, critic, tail)
    opt = ppo.adam_init(trainable)
    counts = np.bincount(data['outcome'][train_rows], minlength=3) + 1e-9
    freq = counts / counts.sum()
    val_freq_ce = float(-np.log(freq[data['outcome'][val_rows]]).mean())
    before = validate(trainable)
    print('validation before:', json.dumps({k: round(v, 4) for k, v in before.items()}))
    best = (before['wdl'], 0, trainable)
    started = time.time()
    for step in range(1, args.steps + 1):
        index = rng.choice(train_rows, args.batch, replace=False)
        (_, metrics), grads = grad(trainable, batch_of(index))
        trainable, opt, _ = ppo.adam_step(trainable, grads, opt, learning_rate=args.lr, max_norm=1.)
        if step % args.log_every == 0 or step == 1:
            print(f'step {step}: loss {float(metrics["loss"]):.4f} wdl {float(metrics["wdl"]):.4f} '
                  f'critic {float(metrics["critic"]):.4f} distill {float(metrics["distill"]):.4f} '
                  f'({time.time() - started:.0f}s)', flush=True)
        if args.val_every and step % args.val_every == 0 and step < args.steps:
            mid = validate(trainable)
            print(f'validation at step {step}:', json.dumps({k: round(v, 4) for k, v in mid.items()}), flush=True)
            if mid['wdl'] < best[0]:
                best = (mid['wdl'], step, trainable)
    after = validate(trainable)
    if args.val_every:
        if after['wdl'] < best[0]:
            best = (after['wdl'], args.steps, trainable)
        print(f'best validation WDL cross-entropy {best[0]:.4f} at step {best[1]} (kept)')
        trainable = best[2]
    print('validation after: ', json.dumps({k: round(v, 4) for k, v in after.items()}))
    print(f'validation baseline: outcome-frequency cross-entropy {val_freq_ce:.4f} '
          f'(train frequencies win/draw/loss {np.round(freq, 3).tolist()})')
    if 'opp_tech' in data:
        print('belief baselines on the validation rows with labels:', json.dumps(_belief_baselines(
            data, val_rows[data['has_belief'][val_rows]], train_rows[data['has_belief'][train_rows]])))
    if args.out:
        herox.save(args.out, merged(trainable[0], trainable[2]), trainable[1],
                   json.loads(Path(args.init).with_suffix('.json').read_text())['base'] if args.init else
                   dict(size=SIZE, blocks=6, width=128, architecture='transformer-v3-p2-h4', obs_version=3),
                   smoke_data=str(args.data), steps=args.steps, stage=args.stage, unfreeze_blocks=n,
                   scoring_only=bool(n), best_step=best[1] if args.val_every else args.steps)
        print('saved', args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--init', default='', help='heroX checkpoint (default: convert weights/plato)')
    c.add_argument('--games', type=int, default=16)
    c.add_argument('--lanes', type=int, default=8, help='games played at once')
    c.add_argument('--max-turns', type=int, default=None, help='round cap (default: the training limit, 200)')
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--seed-base', type=int, default=900_000, help='first map seed')
    c.add_argument('--keep', type=float, default=1., help='fraction of decisions stored (random)')
    c.add_argument('--shard-games', type=int, default=100, help='finished games per data shard')
    c.add_argument('--out', default='runs/herox/smoke-data.npz', help='shards are <stem>-NNN.npz beside it')
    c.set_defaults(run=collect)
    t = sub.add_parser('train')
    t.add_argument('--init', default='')
    t.add_argument('--data', default='runs/herox/smoke-data.npz', help='one dataset, or several separated by commas')
    t.add_argument('--steps', type=int, default=200)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--val-rows', type=int, default=512)
    t.add_argument('--log-every', type=int, default=25)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--out', default='')
    t.add_argument('--stage', choices=('value', 'all'), default='all',
                   help='value: train only the value heads and the critic (FiLM, plan embedding, gate, macro frozen)')
    t.add_argument('--unfreeze-blocks', type=int, default=0,
                   help='also train the last N transformer blocks and the final norm (a scoring copy for search)')
    t.add_argument('--val-every', type=int, default=0, help='validate every N steps and keep the best (WDL) weights')
    t.set_defaults(run=train)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
