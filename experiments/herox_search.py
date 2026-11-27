"""Step 1 of the search plan: Plato + whole-turn search scored by heroX's value token, against raw greedy Plato.

    .venv/bin/python experiments/herox_search.py --arm leakfree --maps 50 --out runs/herox/search-leakfree.json
    .venv/bin/python experiments/herox_search.py --arm leaky    --maps 50 --out runs/herox/search-leaky.json

The candidate generates K candidate turns with Plato's policy (lane 0 = Plato's greedy turn, lanes 1.. sampled at
temperature 1), plays one opponent reply per lane (Plato at temperature 1; depth 1) and scores each lane with the heroX
value token (p(win) - p(loss)) at its own next decision (rl/search.py, rl/search_arena.py). Plato's own policy has no
value head, so without a scorer every lane ties and search keeps the greedy turn.

  leakfree: lanes run on worlds sampled from the searcher's own view (rl/belief.py) and the plan is committed only up
            to the first sub-action whose outcome differs from the sample (determinize=True).
  leaky:    lanes run on the true state (sees through fog): the upper bound a perfect belief model could reach.

Opponent: raw greedy Plato. Raw Plato against itself scores exactly 0.5 on every (map, both seats) pair, since both
sides are the same deterministic policy, so the score of an arm is its gain over raw Plato on those maps.
Same maps for every arm (paired). Elo CIs: bootstrap over maps (search_arena.summarize).
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', choices=('leakfree', 'leaky'), required=True)
    parser.add_argument('--plato', default=str(ROOT / 'weights' / 'plato' / 'final.npz'))
    parser.add_argument('--herox', default='runs/herox/stageA.npz', help='heroX checkpoint whose value token scores')
    parser.add_argument('--maps', type=int, default=50, help='maps; each is played from both seats')
    parser.add_argument('--seed-base', type=int, default=6_000_000)
    parser.add_argument('--lanes', type=int, default=16)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--replies', type=int, default=0, help='>0: robust d=1 with this reply budget')
    parser.add_argument('--processes', type=int, default=12)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    from polytopia_bot.rl import search_arena
    from polytopia_bot.rl.distributed import _env_setup
    cfg = dict(candidate=args.plato, opponent=args.plato, herox_value=str(Path(args.herox).resolve()),
               lanes=args.lanes, depth=args.depth, temperature=1., rollout_temperature=1., replies=args.replies,
               determinize=args.arm == 'leakfree')
    items = [(args.seed_base + m, seat) for m in range(args.maps) for seat in (1, 2)]
    _env_setup()
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    processes = min(args.processes, len(items))
    for _ in range(processes):
        queue.put(None)
    rows, started = [], time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ctx.Pool(processes, initializer=search_arena._init, initargs=(cfg, queue)) as pool:
        for row in pool.imap_unordered(search_arena._play, items):
            rows.append(row)
            result = 'win' if row['reward'] > 0 else 'loss' if row['reward'] < 0 else 'draw'
            print(f"[{len(rows)}/{len(items)} {time.time() - started:.0f}s] map {row['seed']} seat {row['seat']}: "
                  f"{result} in {row['turns']} rounds, {row['search_seconds'] / max(row['search_turns'], 1):.1f} s/turn, "
                  f"overrides {row['overrides']}/{row['plans']}", flush=True)
            out.write_text(json.dumps(dict(cfg=cfg, rows=rows), indent=1) + '\n')
    summary = search_arena.summarize(rows)
    out.write_text(json.dumps(dict(cfg=cfg, summary=summary, rows=rows), indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'unit_mix'}, indent=1))


if __name__ == '__main__':
    main()
