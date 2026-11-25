"""heroX against Plato on engine-rs (CPU): every game played at once, heroX alternating seats.

    .venv/bin/python experiments/herox_arena.py --herox runs/herox/k1000-trained.npz --games 25

Both sides sample their moves (temperature 1, as the repository's Plato benchmarks) from their own view; legal masks
come from the true state, as in training. heroX chooses a plan with its macro head at the start of each of its turns.
Score = (wins + draws / 2) / games for heroX. A game reaching the round cap is a draw here.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--herox', required=True)
    parser.add_argument('--plato', default=str(ROOT / 'weights' / 'plato' / 'final.npz'))
    parser.add_argument('--games', type=int, default=25)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--seed-base', type=int, default=5_000_000, help='first map seed (training used 900,000+ and 1,000,000+)')
    parser.add_argument('--out', default='runs/herox/arena.json')
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import herox, network
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.distributed import _template_params, network_spec
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.rust_engine import RustGames

    params, _ = herox.load(args.herox)
    plato = load_checkpoint(args.plato, _template_params(**network_spec(args.plato)))
    n = args.games
    herox_seat = np.array([1 if i % 2 == 0 else 2 for i in range(n)])
    engine = RustGames(SIZE, n)
    act_x = jax.jit(jax.vmap(lambda f, m, k, z: herox.act(params, f, m, k, z, size=SIZE)[0]))
    act_p = jax.jit(jax.vmap(lambda f, m, k: network.sample(plato, f, m, k, size=SIZE)[0]))
    choose = jax.jit(jax.vmap(lambda f, z, k: herox.choose_plan(params, f, z, k)[0]))
    seeds = np.arange(args.seed_base, args.seed_base + n)
    states, prepared = engine.reset(seeds)
    key = jax.random.key(args.seed)
    plan = np.zeros(n, np.int32)
    last = [None] * n
    results = [None] * n
    started, ticks = time.time(), 0
    while any(r is None for r in results):
        mover, turn = np.asarray(states.current_player), np.asarray(states.turn)
        ours = mover == herox_seat
        key, kp, kx, ko = jax.random.split(key, 4)
        new = np.array([ours[i] and last[i] != int(turn[i]) for i in range(n)])
        if new.any():
            plan = np.where(new, np.asarray(choose(prepared.features, jnp.asarray(plan), jax.random.split(kp, n))), plan)
        for i in np.nonzero(ours)[0]:
            last[i] = int(turn[i])
        a = act_x(prepared.features, prepared.masks, jax.random.split(kx, n), jnp.asarray(plan))
        b = act_p(prepared.features, prepared.masks, jax.random.split(ko, n))
        sel = jnp.asarray(ours)
        choice = Choice(*(jnp.where(sel, x, y) for x, y in zip(a, b)))
        # finished lanes start a new game on the same seed; their moves are ignored
        tick, status = engine.tick(states, choice, seeds)
        ticks += 1
        if np.any(np.asarray(status)[[i for i in range(n) if results[i] is None]] != 0):
            raise RuntimeError('illegal action')
        done, rewards, capped = (np.asarray(v) for v in (tick.done, tick.rewards, tick.truncated))
        for i in np.nonzero(done)[0]:
            if results[i] is not None:
                continue
            r = float(rewards[i][herox_seat[i] - 1])
            outcome = 'draw' if capped[i] else ('win' if r > 0 else 'loss')
            results[i] = dict(game=int(i), map_seed=int(seeds[i]), herox_seat=int(herox_seat[i]), outcome=outcome,
                              rounds=int(turn[i]))
            finished = sum(r is not None for r in results)
            print(f'game {i:2d} (heroX seat {herox_seat[i]}): heroX {outcome:4s} in {int(turn[i])} rounds  '
                  f'[{finished}/{n}, {time.time() - started:.0f}s]', flush=True)
        states, prepared = tick.states, tick.prepared
    tally = {k: sum(r['outcome'] == k for r in results) for k in ('win', 'draw', 'loss')}
    score = (tally['win'] + tally['draw'] / 2) / n
    by_seat = {s: {k: sum(r['outcome'] == k for r in results if r['herox_seat'] == s) for k in ('win', 'draw', 'loss')}
               for s in (1, 2)}
    summary = dict(herox=args.herox, plato=args.plato, games=n, **tally, herox_score=round(score, 3), by_seat=by_seat,
                   seconds=round(time.time() - started, 1), results=results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=1))


if __name__ == '__main__':
    main()
