"""heroX stage B: give the 16 plans a meaning by hindsight labels, then train FiLM, the plan embedding, the query gate
and the macro head to imitate them (value heads and Plato frozen).

    .venv/bin/python experiments/herox_plans.py label --data runs/herox/plans1000.npz
    .venv/bin/python experiments/herox_plans.py train --init runs/herox/stageA.npz --data runs/herox/plans1000.npz \
        --out runs/herox/stageB.npz

label: every turn of the self-play data (herox_smoke.py collect records a summary of EVERY decision) becomes a vector:
the share of each action family (end and resign left out; square roots, as for histograms), the number of actions and
the mean move advance toward the opponent's territory. k-means (k = PLAN_COUNT) on the training games' turns gives
centroids; every turn gets the nearest one as its plan label.

train: herox.plan_loss. Each stored action is imitated under its turn's label (-log pi(a | s, label)); the macro head
learns to predict the label at the first decision of the turn, from a pass conditioned on the seat's previous label.
Trained: herox.HEAD_SIDE_PLAN_HEADS (plan embedding, head FiLM, family bias, query gate, macro head). Input FiLM and
plan_token stay frozen, so the trunk and the value heads are exactly those of the --init checkpoint.
Validation (held-out games), before and after:
  * nll_label: -log pi(a | s, label). Before training it is Plato's own NLL for every label.
  * nll_wrong: the same under a wrong label. A gap to nll_label = the plans change behaviour (controllability).
  * nll_mixture: -log sum_z p_macro(z | turn start) pi(a | s, z). At most Plato's NLL = the macro head and the plans
    together still reproduce Plato (faithfulness).
  * macro cross-entropy and accuracy, against the label-frequency baseline.
  * family table: the policy's mean family distribution under each plan, against that cluster's real family shares.
  * value: WDL cross-entropy and accuracy of the (frozen) value heads under the label, to see what FiLM did to them.
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


def shards(path):
    path = Path(path)
    return sorted(path.parent.glob(f'{path.stem}-[0-9][0-9][0-9].npz')) or [path]


def load(path, keys):
    parts = []
    for f in shards(path):
        with np.load(f) as archive:
            parts.append({k: archive[k] for k in keys})
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def split_games(games, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(games)
    return set(rng.choice(unique, max(1, len(unique) // 5), replace=False).tolist())


def turn_vectors(t, families):
    """Feature vector per turn (see the module docstring)."""
    from polytopia_bot.rl.actions import END, RESIGN
    counts = t['turn_counts'].astype(np.float64)
    acting = [f for f in range(len(families)) if f not in (END, RESIGN)]
    n = counts[:, acting].sum(1)
    share = counts[:, acting] / np.maximum(n, 1)[:, None]
    advance = t['turn_advance'] / np.maximum(t['turn_advance_moves'], 1)
    x = np.concatenate((np.sqrt(share), (.25 * np.log1p(n))[:, None], (.5 * np.tanh(advance))[:, None]), 1)
    return x, [families[f] for f in acting] + ['log_actions', 'advance']


def kmeans(x, k, seed, restarts=5, iterations=100):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centres = [x[rng.integers(len(x))]]
        for _ in range(k - 1):                              # k-means++
            d = np.min([((x - c) ** 2).sum(1) for c in centres], 0)
            centres.append(x[rng.choice(len(x), p=d / d.sum())])
        centres = np.array(centres)
        for _ in range(iterations):
            label = (((x[:, None] - centres[None]) ** 2).sum(-1)).argmin(1)
            new = np.array([x[label == j].mean(0) if (label == j).any() else centres[j] for j in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((x - centres[label]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, centres)
    return best[1]


def assign(x, centres):
    return (((x[:, None] - centres[None]) ** 2).sum(-1)).argmin(1)


def label(args):
    from polytopia_bot.rl import herox
    from polytopia_bot.rl.actions import FAMILIES
    t = load(args.data, ('turn_game', 'turn_seat', 'turn_turn', 'turn_counts', 'turn_advance', 'turn_advance_moves'))
    x, names = turn_vectors(t, FAMILIES)
    val = np.isin(t['turn_game'], list(split_games(t['turn_game'], args.seed)))
    centres = kmeans(x[~val], herox.PLAN_COUNT, args.seed)
    labels = assign(x, centres)
    out = Path(args.data).with_name(Path(args.data).stem + '-labels.npz')
    np.savez(out, game=t['turn_game'], seat=t['turn_seat'], turn=t['turn_turn'], label=labels.astype(np.int32),
             centres=centres, counts=t['turn_counts'])
    shares = t['turn_counts'] / np.maximum(t['turn_counts'].sum(1, keepdims=True), 1)
    advance = t['turn_advance'] / np.maximum(t['turn_advance_moves'], 1)
    clusters = []
    for j in range(herox.PLAN_COUNT):
        m = labels == j
        mean = shares[m].mean(0)
        top = {FAMILIES[f]: round(float(mean[f]), 2) for f in np.argsort(-mean)[:4]}
        clusters.append(dict(plan=j, turns=int(m.sum()), mean_round=round(float(t['turn_turn'][m].mean()), 1),
                             actions=round(float(t['turn_counts'][m].sum(1).mean()), 1),
                             advance=round(float(advance[m].mean()), 2), top_families=top))
        print(json.dumps(clusters[-1]))
    print(f'{len(labels)} turns from {len(np.unique(t["turn_game"]))} games; features {names}; saved {out}')
    out.with_suffix('.json').write_text(json.dumps(dict(features=names, clusters=clusters), indent=1) + '\n')


def train(args):
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import herox, ppo
    from polytopia_bot.rl.actions import FAMILIES, Choice, PathMasks
    from polytopia_bot.rl.cpu_engine import view_features, view_layout

    params, critic = herox.load(args.init)
    keys = ('view', 'family', 'source', 'target', 'family_mask', 'source_mask', 'target_mask', 'game', 'seat', 'turn',
            'first', 'outcome')
    data = load(args.data, keys)
    lab = np.load(Path(args.data).with_name(Path(args.data).stem + '-labels.npz'))
    table = {(int(g), int(s), int(t)): int(l) for g, s, t, l in zip(lab['game'], lab['seat'], lab['turn'], lab['label'])}
    data['plan'] = np.array([table[(int(g), int(s), int(t))] for g, s, t in zip(data['game'], data['seat'], data['turn'])],
                            np.int32)
    data['previous'] = np.array([table.get((int(g), int(s), int(t) - 1), 0)
                                 for g, s, t in zip(data['game'], data['seat'], data['turn'])], np.int32)
    val_games = split_games(data['game'], args.seed)
    is_val = np.isin(data['game'], list(val_games))
    first = data['first'].astype(bool)
    rng = np.random.default_rng(args.seed)
    train_rows, train_starts = np.nonzero(~is_val)[0], np.nonzero(~is_val & first)[0]
    val_rows = rng.choice(np.nonzero(is_val)[0], min(args.val_rows, int(is_val.sum())), replace=False)
    val_starts = np.nonzero(is_val & first)[0]
    val_starts = rng.choice(val_starts, min(args.val_rows, len(val_starts)), replace=False)
    print(f'{len(data["game"])} rows ({first.sum()} turn starts) from {len(np.unique(data["game"]))} games; '
          f'train {len(train_rows)} rows / {len(train_starts)} starts, validation {len(val_rows)} / {len(val_starts)}')
    layout = view_layout(SIZE, 2)
    features = jax.jit(jax.vmap(lambda v: view_features(v, layout, SIZE)))

    def rows_of(index, plan=None):
        f = features(jnp.asarray(data['view'][index]))
        return dict(board=f.board, global_state=f.global_state,
                    choice=Choice(*(jnp.asarray(data[k][index]) for k in ('family', 'source', 'target'))),
                    masks=PathMasks(*(jnp.asarray(data[k][index]) for k in ('family_mask', 'source_mask', 'target_mask'))),
                    plan=jnp.asarray(data['plan'][index] if plan is None else plan),
                    previous=jnp.asarray(data['previous'][index]), label=jnp.asarray(data['plan'][index]),
                    outcome=jnp.asarray(data['outcome'][index]))

    train_heads, frozen = herox.head_group(params['heads'], herox.HEAD_SIDE_PLAN_HEADS)
    merged = lambda h: dict(params, heads={**frozen, **h})

    def loss(heads, rows, starts):
        return herox.plan_loss(merged(heads), rows, starts, macro_coef=args.macro_coef)

    grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    logp_of = jax.jit(lambda h, b: jax.vmap(herox.action_logp, in_axes=(None, 0, 0, 0, 0, 0))(
        merged(h), b['board'], b['global_state'], b['choice'], b['masks'], b['plan']))
    every_plan = jnp.arange(herox.PLAN_COUNT)
    logp_all = jax.jit(lambda h, b: jax.vmap(lambda z: jax.vmap(herox.action_logp, in_axes=(None, 0, 0, 0, 0, None))(
        merged(h), b['board'], b['global_state'], b['choice'], b['masks'], z))(every_plan).T)
    macro_of = jax.jit(lambda h, b: jax.vmap(lambda bd, g, z: herox.macro_logits(merged(h), herox.encode(merged(h), bd, g, z)))(
        b['board'], b['global_state'], b['previous']))
    families_all = jax.jit(lambda h, b: jax.vmap(lambda z: jax.vmap(herox.family_probs, in_axes=(None, 0, 0, 0, None))(
        merged(h), b['board'], b['global_state'], b['masks'].family, z).mean(0))(every_plan))
    wdl_of = jax.jit(lambda h, b: jax.vmap(lambda bd, g, z: herox.value_heads(merged(h), herox.encode(merged(h), bd, g, z))['wdl'])(
        b['board'], b['global_state'], b['plan']))

    # turn starts of the validation games, for the mixture: (game, seat, turn) -> row of that turn's first decision
    start_of = {(int(data['game'][i]), int(data['seat'][i]), int(data['turn'][i])): i for i in np.nonzero(is_val & first)[0]}
    mix_rows = np.array([i for i in val_rows if (int(data['game'][i]), int(data['seat'][i]), int(data['turn'][i])) in start_of]
                        )[:args.mix_rows]
    label_freq = np.bincount(data['plan'][train_starts], minlength=herox.PLAN_COUNT) + 1e-9
    label_freq /= label_freq.sum()
    centre_shares = []
    for j in range(herox.PLAN_COUNT):
        c = lab['counts'][lab['label'] == j].astype(np.float64)
        centre_shares.append((c / np.maximum(c.sum(1, keepdims=True), 1)).mean(0))
    centre_shares = np.array(centre_shares)

    def batches(index, size=128):
        return [index[i:i + size] for i in range(0, len(index), size)]

    def validate(heads, label):
        out = {}
        nll = np.concatenate([-np.asarray(logp_of(heads, rows_of(b))) for b in batches(val_rows)])
        wrong = (data['plan'][val_rows] + rng.integers(1, herox.PLAN_COUNT, len(val_rows))) % herox.PLAN_COUNT
        nll_wrong = np.concatenate([-np.asarray(logp_of(heads, rows_of(b, wrong[i:i + len(b)])))
                                    for i, b in zip(range(0, len(val_rows), 128), batches(val_rows))])
        out.update(nll_label=nll.mean(), nll_wrong=nll_wrong.mean())
        mixture = []
        for b in batches(mix_rows, 64):
            lp = np.asarray(logp_all(heads, rows_of(b)))                            # [rows, plans]
            starts = np.array([start_of[(int(data['game'][i]), int(data['seat'][i]), int(data['turn'][i]))] for i in b])
            lq = np.asarray(jax.nn.log_softmax(macro_of(heads, rows_of(starts))))
            mixture.append(-np.asarray(jax.scipy.special.logsumexp(lp + lq, axis=1)))
        mixture = np.concatenate(mixture)
        nll_mix_rows = np.concatenate([-np.asarray(logp_of(heads, rows_of(b))) for b in batches(mix_rows)])
        out.update(nll_mixture=mixture.mean(), nll_label_same_rows=nll_mix_rows.mean())
        macro = np.concatenate([np.asarray(macro_of(heads, rows_of(b))) for b in batches(val_starts)])
        target = data['plan'][val_starts]
        logq = np.asarray(jax.nn.log_softmax(macro))
        out.update(macro_ce=-logq[np.arange(len(target)), target].mean(),
                   macro_accuracy=(macro.argmax(1) == target).mean(),
                   macro_baseline_ce=-np.log(label_freq[target]).mean())
        wdl = np.concatenate([np.asarray(wdl_of(heads, rows_of(b))) for b in batches(val_rows)])
        outcome = data['outcome'][val_rows]
        lw = np.asarray(jax.nn.log_softmax(wdl))
        out.update(value_wdl_ce=-lw[np.arange(len(outcome)), outcome].mean(), value_accuracy=(wdl.argmax(1) == outcome).mean())
        fam = np.asarray(families_all(heads, rows_of(val_rows[:args.family_rows])))      # [plans, families]
        corr = [np.corrcoef(fam[j], centre_shares[j])[0, 1] for j in range(herox.PLAN_COUNT)]
        spread = np.abs(fam - fam.mean(0)).sum(1).mean() / 2                             # mean TV from the plan average
        out.update(family_centre_corr=float(np.nanmean(corr)), family_spread_tv=float(spread))
        out = {k: round(float(v), 4) for k, v in out.items()}
        print(f'validation {label}:', json.dumps(out), flush=True)
        return out, fam

    before, _ = validate(train_heads, 'before')
    opt = ppo.adam_init(train_heads)
    started = time.time()
    for step in range(1, args.steps + 1):
        rows = rows_of(rng.choice(train_rows, args.batch, replace=False))
        starts = rows_of(rng.choice(train_starts, args.batch, replace=False))
        (_, m), grads = grad(train_heads, rows, starts)
        train_heads, opt, _ = ppo.adam_step(train_heads, grads, opt, learning_rate=args.lr, max_norm=1.)
        if step % args.log_every == 0 or step == 1:
            print(f'step {step}: bc {float(m["bc"]):.4f} macro {float(m["macro"]):.4f} '
                  f'macro_acc {float(m["macro_accuracy"]):.3f} gate {float(train_heads["plan_query_gate"]):+.4f} '
                  f'({time.time() - started:.0f}s)', flush=True)
    after, fam = validate(train_heads, 'after')
    top = [', '.join(f'{FAMILIES[f]} {fam[j, f]:.2f}/{centre_shares[j, f]:.2f}' for f in np.argsort(-centre_shares[j])[:3])
           for j in range(herox.PLAN_COUNT)]
    print('plan: policy family share / cluster family share (top 3 families of the cluster)')
    for j, t in enumerate(top):
        print(f'  {j:2d}: {t}')
    if args.out:
        info = json.loads(Path(args.init).with_suffix('.json').read_text())
        herox.save(args.out, merged(train_heads), critic, info['base'], init=args.init, data=args.data, steps=args.steps,
                   stage='plans', validation_before=before, validation_after=after)
        Path(args.out).with_name(Path(args.out).stem + '-families.json').write_text(json.dumps(dict(
            policy=fam.round(4).tolist(), clusters=centre_shares.round(4).tolist(), families=list(FAMILIES))) + '\n')
        print('saved', args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('label')
    a.add_argument('--data', required=True)
    a.add_argument('--seed', type=int, default=0)
    a.set_defaults(run=label)
    t = sub.add_parser('train')
    t.add_argument('--init', required=True, help='heroX checkpoint (stage A output)')
    t.add_argument('--data', required=True)
    t.add_argument('--steps', type=int, default=3000)
    t.add_argument('--batch', type=int, default=64, help='rows and turn starts per step (each)')
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--macro-coef', type=float, default=1.)
    t.add_argument('--val-rows', type=int, default=2048)
    t.add_argument('--mix-rows', type=int, default=512)
    t.add_argument('--family-rows', type=int, default=512)
    t.add_argument('--log-every', type=int, default=250)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--out', default='')
    t.set_defaults(run=train)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
