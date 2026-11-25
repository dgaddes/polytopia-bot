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
  ├─ GLOBAL, tiles → family / source / target heads  (Plato's; target query + gate·e_z)
  ├─ PLAN  → LayerNorm → Linear 128→16               macro logits
  ├─ VALUE → WDL (LN → 128→128 → SiLU → 128→3), score margin (21 buckets), turns remaining (20 buckets)
  └─ tiles → ownership (tanh per tile)
privileged critic: Plato's branch on the 50 public tokens + opponent view → 3 WDL logits (training only)
```

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
| Macro head | 2,320 |
| WDL head | 17,155 |
| Score-margin + turns-remaining + ownership heads | 21,930 |
| **heroX policy** | **3,289,656** |
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

## Not built yet

- **Data collection** for `head_loss`: outcomes, score margins, turns remaining and final ownership per position,
  from Plato self-play.
- **PPO integration** (`rl/ppo.py`, `rl/gpu_resident.py`): the macro head's per-turn advantage, the plan stored
  with each row, and the WDL critic's scalar value for GAE.
- **Plan pseudo-rewards** and plan clustering.
- **Test-time search** over plans and sampled worlds using `act`'s `value` output.
- Studio and arena support for heroX checkpoints.

## Usage

```bash
.venv/bin/python scripts/herox.py convert --out runs/herox/init.npz   # from weights/plato/final*.npz
.venv/bin/python scripts/herox.py info runs/herox/init.npz
.venv/bin/python -m pytest tests/test_herox.py
```
