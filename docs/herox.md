# heroX: plan token, value token and WDL critic on Plato

heroX extends Plato (`transformer-v3-p2-h4`, 6×128) with three things:

- a **plan (macro) token** that chooses one of 16 strategies per turn and conditions every action that turn;
- a **value token** with win/draw/loss, score-margin, turns-remaining and ownership heads, read from the
  player's own view (usable at play time, unlike the privileged critic);
- a **WDL privileged critic**, replacing the scalar critic output.

Code: [`src/polytopia_bot/rl/herox.py`](../src/polytopia_bot/rl/herox.py). Tests:
[`tests/test_herox.py`](../tests/test_herox.py). Conversion: [`scripts/herox.py`](../scripts/herox.py).

## Architecture

```
board [14,14,151] + global [525]
  conv stem + 4 residual conv blocks                 (Plato, unchanged)
  2×2 patches → 49 tokens; + GLOBAL                  (Plato, unchanged)
  + PLAN, VALUE tokens                               → 52 tokens
  FiLM(e_z): x ← x·(1+γ) + β on every token          (zero-initialized)
  6 transformer blocks                               (Plato's weights; readout mask, below)
  ├─ GLOBAL, tiles → head FiLM(e_z) → family / source / target heads
  │                  (Plato's heads; family logits + family_bias[z]; target query + gate·e_z)
  ├─ PLAN  → LayerNorm → Linear 128→16               macro logits
  ├─ VALUE → WDL (LN → 128→128 → SiLU → 128→3), score margin (21 buckets), turns remaining (20 buckets)
  └─ tiles → ownership (tanh per tile)
privileged critic: Plato's branch on the 50 public tokens + opponent view → 3 WDL logits (training only)
```

**Head-side plan conditioning.** `head_film` (128→256, zero-initialized) scales and shifts `g` and `tiles` before
the action heads only, and `family_bias` (16×17, zero) shifts the family logits per plan. The value and ownership heads
read the unmodulated outputs. Stage B trains these instead of input FiLM (below).

**Readout mask.** PLAN and VALUE attend to all 52 tokens, but Plato's 50 tokens cannot attend to them: those
keys get a −1e9 attention bias. This is a deliberate change from full attention, so the new tokens can never alter
what Plato's tokens compute. The plan reaches the policy only through FiLM and the gated target-query term.

**Exact start from Plato.** FiLM and `plan_query_gate` start at zero, so `from_plato` gives a network that chooses
Plato's moves with Plato's probabilities for every plan (tested on the released weights to 1e-4; softmax sums over
52 instead of 50 keys round slightly differently). `plan_embedding` starts small and random: if it were zero too,
FiLM and the embedding would block each other's gradients.

**Choosing a plan.** At the first action of a turn, `choose_plan` reads the macro head from a pass conditioned on
the previous plan (0 at game start) and samples the new plan; `act` then uses it for every action of the turn.

**Critic conversion.** Plato's scalar output v becomes logits (win, draw, loss) = (v, −4, −v), so
p(win) − p(loss) ≈ tanh(v). It needs a short warm-up against real outcomes.

## Parameters (14×14, width 128)

| Part | Parameters |
|---|---|
| Plato policy (unchanged) | 3,212,922 |
| PLAN + VALUE tokens | 256 |
| Plan embedding (16×128) + FiLM (128→256) + query gate | 37,121 |
| Head FiLM (128→256) + family bias (16×17) | 33,296 |
| Macro head | 2,320 |
| WDL head | 17,155 |
| Score-margin + turns-remaining + ownership heads | 21,930 |
| **heroX policy** | **3,322,952** |
| WDL critic (training only) | 401,063 |

## Losses available now

`herox.head_loss` trains the new heads with supervised targets:

| Term | Target |
|---|---|
| `wdl` | cross-entropy to the game outcome; a game reaching the round cap is a **draw** here (PPO still scores it −1 for both) |
| `distill` | KL from the critic's WDL (stop-gradient) to the value token's |
| `critic` | the critic's cross-entropy to the outcome |
| `score`, `turns` | cross-entropy to the final score-margin bucket (1,000 points each, ±10,000) and turns-remaining bucket (10 turns each) |
| `ownership` | mean squared error to the final owner of each tile (+1 own, 0 none, −1 opponent) |
| `macro` | cross-entropy to a plan label (e.g. clustered next-turns statistics); rows with label −1 are skipped |

For step 1 of the plan (frozen trunk), differentiate with respect to `params['heads']` only. A test checks that
this leaves Plato's weights bit-identical.

## Frozen-trunk training (stages A and B, 2026-11-26)

Data: Plato self-play through heroX on CPU (`experiments/herox_smoke.py collect`): positions sampled per decision,
the first decision of every turn always kept, and a summary of every turn (action count per family, move advance
toward the opponent's territory). Validation is always on held-out games.

**Stage A: value heads only** (`herox_smoke.py train --stage value`; FiLM, plan embedding, gate, macro and Plato
bit-identical afterwards, so the policy is still exactly Plato's). 1,000 games (228,794 positions), 3,000 steps × 64,
CPU 8 min:

| Validation (200 games) | Before | After |
|---|---|---|
| WDL cross-entropy (outcome-frequency baseline 0.962) | 1.102 | **0.200** |
| WDL accuracy | 6% | **91.7%** |
| Critic WDL cross-entropy | 0.766 | 0.218 |
| Score / turns cross-entropy | 3.04 / 3.00 | 1.49 / 1.17 |
| Ownership MSE | 0.625 | 0.426 |

**Stage B: hindsight plans** (`experiments/herox_plans.py`). New 1,000 games (75,042 turns); k-means on the turn
summaries gives 16 recognizable modes: openings (plans 1, 12), economy (2, 8, 13: forest, harvest, research), large
offensives (3, 5, 9), holding/passive turns (4, 6, 7, 14), and the stalemate shuffle of capped games (15, round 94).
Then `herox.plan_loss` (-log π(a | s, label) plus macro cross-entropy), 3,000 steps × (64 rows + 64 turn starts),
lr 1e-3, CPU 7 min.

| Validation (200 games) | Before | After |
|---|---|---|
| NLL per action under the turn's label (Plato: 2.373) | 2.373 | 2.375 |
| NLL under a wrong label | 2.373 | **2.409** |
| NLL of the macro-weighted mixture over plans (same rows: label 2.378) | 2.377 | 2.379 |
| Macro cross-entropy (label-frequency baseline 2.532) | 2.773 | **1.013** |
| Macro accuracy | 4% | **65%** |
| Value WDL cross-entropy | 0.259 | **0.259** (unchanged, by construction) |

Readings:

- **Faithful.** Sampling the plan from the macro head reproduces Plato (+0.003 nats per action).
- **The macro head works**: it predicts the mode of the coming turn from its first position, 65% of the time.
- **Plans steer behaviour only weakly.** A wrong plan costs 0.034 nats per action. Forcing a plan moves the family
  distribution by 1–13% (total variation) on the same positions: the holding plan 4 halves attacks (0.134 → 0.067)
  and adds moves, plans 1, 6 and 7 move it 9–10%, while the offensive and economy plans stay within 2–4% of Plato.
  The labels are mostly predictable from the position, which Plato already sees, so imitating Plato's own play can
  only teach the plans the little that the position does not decide.
- **Input FiLM was the wrong lever.** The first stage-B run trained input FiLM (before the six frozen pre-norm
  blocks): γ reached 0.5 but the family distribution moved 0.6% (TV), and value cross-entropy rose 0.259 → 0.304
  because FiLM changes every token. That run is why stage B trains the head-side parameters (`HEAD_SIDE_PLAN_HEADS`).

To make plans matter, they need a signal tied to outcomes (plan-level search with the value token, or PPO on the
plan parameters), or more diverse behaviour to imitate (several Plato snapshots, temperatures, multi-turn labels).

## Search scored by the value token (2026-11-27)

Plato's policy has no value head (`network.trunk` returns 0 for transformer-v3), so `TurnSearcher` on Plato alone
scores every lane 0 and keeps the greedy turn, except for lanes that end the game. With `value_params` = a heroX
checkpoint, `search._value` scores lanes with the value token's p(win) − p(loss) instead.

`experiments/herox_search.py`: Plato + plain d=1 search (16 lanes, temperature 1, one opponent reply at temperature 1,
scored at the searcher's next decision by stage A's value token) against raw greedy Plato, 100 maps (seeds
6,000,000+) × both seats, the same games for both arms. Raw Plato vs itself is exactly 0.5 on these pairs.

| Arm | W–L–D | Score | Elo vs raw Plato | s per searched turn (7 games on 14 cores) |
|---|---|---|---|---|
| leak-free (`determinize=True`) | 120–78–2 | 0.605 [0.562, 0.650] | **+74 [+44, +108]** | 4.9 |
| leaky (true state: perfect belief) | 117–81–2 | 0.590 [0.537, 0.640] | +63 [+26, +100] | 3.6 |
| **leaky − leak-free (paired by map)** | | | **−11 [−56, +35]** | |

- The frozen-trunk value token makes search work on Plato: +74 Elo, in the range of the older bots (+56 to +104).
- **A perfect belief adds nothing measurable here.** The older bots lost 58–148 Elo to the crude sampler (most of it
  with robust search, which averages 64 replies, and on classic maps); for Plato with plain d=1 the gap is −11, with
  a CI excluding a gain above +35. A learned belief network is not worth building for this search; recheck if robust
  search becomes the default (its gain depended most on real replies).

### A scoring copy with the last two blocks unfrozen

Search uses Plato for the candidate turns and the heroX checkpoint only to score them, so the scorer's trunk can be
fine-tuned for value without touching the policy search samples from. `herox_smoke.py train --unfreeze-blocks 2`
trains the last 2 of 6 transformer blocks and the final norm with the value heads and critic (the saved checkpoint
is marked `scoring_only`: its policy is no longer Plato's). From stage A, on both datasets (2,000 games, 435,214
positions), 8,000 steps × 64, lr 1e-4, validation every 1,000 steps on 4,096 positions of 400 held-out games
(best kept: the last), CPU ~35 min:

| Validation | Stage A | Unfrozen 2 blocks |
|---|---|---|
| WDL cross-entropy | 0.2153 | **0.2059** |
| WDL accuracy | 90.8% | 91.4% |
| Ownership MSE / score CE / turns CE | 0.432 / 1.547 / 1.215 | 0.362 / 1.483 / 1.096 |

Search with it, same 100 maps × both seats as above (`--herox runs/herox/scorer-u2.npz`):

| Scorer | W–L–D | Elo vs raw Plato | s per searched turn |
|---|---|---|---|
| stage A | 120–78–2 | +74 [+44, +108] | 4.9 |
| **unfrozen 2 blocks** | 128–69–3 | **+106 [+69, +145]** | 5.5 |
| **difference (paired by map)** | better on 20 games, worse on 11 | **+32 [−9, +70]** (positive in 93% of resamples) | |

A 4% lower WDL cross-entropy is worth about +30 Elo in search: the scorer's quality is the lever, not the belief.
Not significant at 100 maps; more data and more unfrozen blocks are the next steps.

## Not built yet

- **PPO integration** (`rl/ppo.py`, `rl/gpu_resident.py`): the macro head's per-turn advantage, the plan stored
  with each row, and the WDL critic's scalar value for GAE.
- **Plan pseudo-rewards**; multi-turn plan labels.
- **Test-time search** over plans and sampled worlds using `act`'s `value` output.
- Studio and arena support for heroX checkpoints.

## Usage

```bash
.venv/bin/python scripts/herox.py convert --out runs/herox/init.npz   # from weights/plato/final*.npz
.venv/bin/python scripts/herox.py info runs/herox/init.npz
.venv/bin/python -m pytest tests/test_herox.py

# stage A, then stage B
.venv/bin/python experiments/herox_smoke.py collect --init runs/herox/init.npz --games 1000 --lanes 32 --keep 0.125 \
    --out runs/herox/k1000.npz
.venv/bin/python experiments/herox_smoke.py train --stage value --init runs/herox/init.npz --data runs/herox/k1000.npz \
    --steps 3000 --batch 64 --val-rows 4096 --out runs/herox/stageA.npz
.venv/bin/python experiments/herox_smoke.py collect --init runs/herox/init.npz --games 1000 --lanes 32 --keep 0.08 \
    --seed-base 2000000 --out runs/herox/plans1000.npz
.venv/bin/python experiments/herox_plans.py label --data runs/herox/plans1000.npz
.venv/bin/python experiments/herox_plans.py train --init runs/herox/stageA.npz --data runs/herox/plans1000.npz \
    --steps 3000 --lr 1e-3 --out runs/herox/stageB.npz
```
