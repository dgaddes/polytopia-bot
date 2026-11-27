"""heroX: Plato's hybrid V3 network plus a plan (macro) token, a value token and a win/draw/loss critic.

Layout (transformer-v3-p2-h4 at 14x14, width 128; docs/herox.md has the full design):

  * Tokens: 49 patch tokens + GLOBAL (Plato's 50) + PLAN + VALUE = 52.
  * PLAN and VALUE are READOUT tokens. They attend to all 52 tokens, but Plato's 50 tokens cannot attend to them (an
    additive -1e9 bias on those keys). So the new tokens never change what the original tokens compute; the plan
    reaches the policy only through FiLM and the target query, below.
  * Plan conditioning: plan z in [0, PLAN_COUNT) selects a row e_z of `plan_embedding`. FiLM scales and shifts every
    token before block 1 (x <- x * (1 + gamma(e_z)) + beta(e_z)), and plan_query_gate * e_z is added to the target
    head's query.
  * Macro head: LayerNorm -> Linear on the PLAN output, giving logits over PLAN_COUNT plans.
  * Value heads on the VALUE output: WDL (p(win), p(draw), p(loss)), final score margin (SCORE_BUCKETS buckets),
    turns remaining (TURN_BUCKETS buckets). An ownership head (tanh, per tile) reads the tile features.
  * Critic: Plato's privileged critic with its scalar output replaced by WDL logits. It reads only the 50 public
    tokens (never PLAN or VALUE) plus the other seat's view. Training only.

Zero initialization of FiLM and of plan_query_gate (plan_embedding itself is small and random, so FiLM's weights get
gradients from the first step), together with the readout mask, means a network made by from_plato
plays exactly as Plato does (up to float rounding in the softmax sums) for every plan, until training moves those
weights. The macro choice is sequential: the PLAN token is read from a pass conditioned on the plan in force (the
previous turn's plan; 0 at the start of a game), a new plan is sampled from it at the first action of a turn, and that
plan conditions every action of the turn.

Not included here: PPO integration of the macro head (per-turn advantages), plan pseudo-rewards, and test-time search.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import network, transformer_v3
from .network import layer_norm
from .observation import fit_board, fit_global
from .transformer import REMATERIALIZE, _checkpointed, block, linear

ARCHITECTURE = 'herox-v1'
PLAN_COUNT = 16
WIN, DRAW, LOSS = 0, 1, 2
SCORE_BUCKETS, SCORE_SCALE = 21, 1000.   # margin buckets of 1,000 points, clipped to [-10,000, +10,000]
TURN_BUCKETS, TURN_SCALE = 20, 10        # turns remaining in buckets of 10, the last bucket open (190+)
READOUT_TOKENS = 2                       # PLAN, VALUE, after GLOBAL
MASKED = -1e9
DRAW_BIAS = -4.                          # from_plato's critic: p(draw) starts small, p(win) - p(loss) ~ tanh(value)
# Opponent-state (belief) heads on the VALUE token: auxiliary targets from the true state, inputs stay the own view.
TECH_SLOTS = 128                         # players.tech width; the loss counts only the real tech ids (tech_mask)
ECONOMY = ('stars', 'income', 'cities', 'city_levels', 'units')
ECONOMY_BUCKETS = 16
ECONOMY_SCALE = dict(stars=5, income=2, cities=1, city_levels=3, units=2)   # bucket width; the last bucket is open


# ---------------------------------------------------------------- parameters

def init_heads(key, width: int, families: int = 17) -> dict:
    """The new parameters (196,766 at width 128). FiLM, head FiLM, the family bias and plan_query_gate start at zero,
    so the policy is unchanged; plan_embedding is small and random (both zero would leave each other without a
    gradient)."""
    keys = iter(jax.random.split(key, 14))

    def dense(a, b, scale=1.):
        return dict(w=jax.random.normal(next(keys), (a, b)) * (scale / a ** .5), b=jnp.zeros(b))

    def norm():
        return dict(scale=jnp.ones(width), bias=jnp.zeros(width))

    return dict(
        plan_token=jax.random.normal(next(keys), (width,)) * .02,
        value_token=jax.random.normal(next(keys), (width,)) * .02,
        plan_embedding=jax.random.normal(next(keys), (PLAN_COUNT, width)) * .1,
        plan_query_gate=jnp.zeros(()),
        film=dict(w=jnp.zeros((width, 2 * width)), b=jnp.zeros(2 * width)),
        head_film=dict(w=jnp.zeros((width, 2 * width)), b=jnp.zeros(2 * width)),
        family_bias=jnp.zeros((PLAN_COUNT, families)),
        macro_norm=norm(), macro=dense(width, PLAN_COUNT, .01),
        wdl_norm=norm(), wdl_hidden=dense(width, width), wdl=dense(width, 3, .01),
        score_hidden=dense(width, width), score=dense(width, SCORE_BUCKETS, .01),
        turns=dense(width, TURN_BUCKETS, .01),
        ownership=dense(width, 1, .01),
        belief_norm=norm(), belief_hidden=dense(width, 2 * width),
        belief_tech=dense(2 * width, TECH_SLOTS, .01), belief_economy=dense(2 * width, len(ECONOMY) * ECONOMY_BUCKETS, .01),
    )


def critic_from_v3(critic: dict) -> dict:
    """A V3 critic with its scalar `value` layer replaced by WDL logits: win = v, draw = DRAW_BIAS, loss = -v, so
    p(win) - p(loss) = (e^v - e^-v) / (e^v + e^-v + e^DRAW_BIAS), close to tanh(v). Needs a short critic warm-up."""
    w, b = critic['value']['w'], critic['value']['b']
    wdl = dict(w=jnp.concatenate((w, jnp.zeros_like(w), -w), axis=1),
               b=jnp.concatenate((b, jnp.full_like(b, DRAW_BIAS), -b)))
    return {**{k: v for k, v in critic.items() if k != 'value'}, 'wdl': wdl}


def from_plato(policy: dict, critic: dict, key=None) -> tuple:
    """heroX (params, critic) from a transformer-v3 policy and its critic, e.g. weights/plato/final*.npz."""
    if 'v3_stem' not in policy:
        raise ValueError('from_plato needs a transformer-v3 policy')
    width = policy['global_token']['w'].shape[1]
    key = jax.random.key(0) if key is None else key
    families = policy['family']['w'].shape[1]
    return dict(policy=policy, heads=init_heads(key, width, families)), critic_from_v3(critic)


def init(key, board_channels: int, global_channels: int, *, size: int, width: int, blocks: int,
         architecture: str = 'transformer-v3-p2-h4') -> tuple:
    """A fresh heroX (params, critic) on a transformer-v3 base."""
    a, b, c = jax.random.split(key, 3)
    policy = transformer_v3.init_policy(a, board_channels, global_channels, size=size, width=width, blocks=blocks,
                                        architecture=architecture)
    critic = transformer_v3.init_critic(b, board_channels, global_channels, size=size, width=width,
                                        architecture=architecture)
    return from_plato(policy, critic, c)


BELIEF_HEADS = ('belief_norm', 'belief_hidden', 'belief_tech', 'belief_economy')
VALUE_HEADS = ('value_token', 'wdl_norm', 'wdl_hidden', 'wdl', 'score_hidden', 'score', 'turns', 'ownership') + BELIEF_HEADS
PLAN_HEADS = ('plan_token', 'plan_embedding', 'plan_query_gate', 'film', 'head_film', 'family_bias', 'macro_norm', 'macro')
# Stage B (frozen trunk): the plan acts on the policy heads only. Input FiLM and plan_token stay as they are, so every
# token, and so the value heads, compute exactly what they did after stage A. (Input FiLM through six frozen pre-norm
# blocks moved the policy very little: docs/herox.md.)
HEAD_SIDE_PLAN_HEADS = ('plan_embedding', 'plan_query_gate', 'head_film', 'family_bias', 'macro_norm', 'macro')


def head_group(heads: dict, names) -> tuple:
    """(the heads in `names`, the rest): train one group with the other frozen, merging them back with {**a, **b}."""
    return {k: v for k, v in heads.items() if k in names}, {k: v for k, v in heads.items() if k not in names}


def parameter_counts(params: dict, critic: dict) -> dict:
    count = lambda tree: int(sum(x.size for x in jax.tree.leaves(tree)))
    return dict(base_policy=count(params['policy']), new_heads=count(params['heads']), policy=count(params),
                critic=count(critic))


# ---------------------------------------------------------------- forward pass

def attention_bias(table: jax.Array, grid: int) -> jax.Array:
    """[heads, T, T] for T = grid^2 + 1 + READOUT_TOKENS: Plato's relative bias between patches, zero for the
    GLOBAL / PLAN / VALUE rows and columns, and MASKED where a patch or GLOBAL query meets a PLAN or VALUE key."""
    n = grid * grid
    public = n + 1
    total = public + READOUT_TOKENS
    tiles = jnp.arange(n)
    row, col = tiles // grid, tiles % grid
    rel = table[:, row[:, None] - row[None, :] + grid - 1, col[:, None] - col[None, :] + grid - 1]
    bias = jnp.pad(rel, ((0, 0), (0, total - n), (0, total - n)))
    blocked = (jnp.arange(total)[:, None] < public) & (jnp.arange(total)[None, :] >= public)
    return jnp.where(blocked[None], MASKED, bias)


def encode(params: dict, board: jax.Array, global_state: jax.Array, plan) -> dict:
    """Public encoder on the mover's own view, conditioned on `plan` (an int in [0, PLAN_COUNT)).

    Returns tokens (the 50 public tokens before the final norm, for the critic), tiles [size^2, W], g, the PLAN and
    VALUE outputs, and e (the plan embedding)."""
    policy, heads = params['policy'], params['heads']
    size = board.shape[0]
    patch = transformer_v3._patch_of(policy)
    grid = transformer_v3._grid(size, patch)
    board = fit_board(board, policy['v3_stem']['w'].shape[-2])
    glob = fit_global(global_state, policy['global_token']['w'].shape[0])
    tokens, tile_embedding = transformer_v3._stem(policy['v3_stem'], policy.get('v3_patch'), board, size, patch,
                                                  policy.get('v3_conv_blocks', ()))
    context = linear(glob, policy['global_token'])
    x = jnp.concatenate((tokens, context[None], heads['plan_token'][None], heads['value_token'][None]), axis=0)
    e = heads['plan_embedding'][plan]
    gamma, beta = jnp.split(network.dense(e, heads['film']), 2)
    x = x * (1. + gamma) + beta
    step = _checkpointed(block) if REMATERIALIZE else block
    for weights in policy['transformer']:
        x = step(x, weights, attention_bias(weights['relative_bias'], grid))
    n = policy['v3_final_norm']
    out = layer_norm(x) * n['scale'] + n['bias']
    patches = grid * grid
    if patch == 1:
        tiles = out[:patches]
    else:
        tiles = transformer_v3._from_patches(linear(out[:patches], policy['v3_unpatch']), size, patch) + tile_embedding
    return dict(tokens=x[:patches + 1], tiles=jax.nn.silu(tiles), g=out[patches], plan=out[patches + 1],
                value=out[patches + 2], e=e, z=plan)


def _conditioned(params: dict, encoded: dict) -> tuple:
    """(policy parameters, tiles, g) for the action heads under the plan:
      * head FiLM: tiles and g scaled and shifted by (1 + gamma, beta) of e_z (the value heads read the unmodulated
        outputs);
      * family_bias[z] added to the family logits;
      * gate * e_z added to the target query: network.target_logits adds family_embedding[family] to the query, so
        shifting every row by it adds it for whichever family is chosen."""
    policy, heads, e = params['policy'], params['heads'], encoded['e']
    gamma, beta = jnp.split(network.dense(e, heads['head_film']), 2)
    family = dict(policy['family'], b=policy['family']['b'] + heads['family_bias'][encoded['z']])
    conditioned = dict(policy, family=family, family_embedding=policy['family_embedding'] + heads['plan_query_gate'] * e)
    return conditioned, encoded['tiles'] * (1. + gamma) + beta, encoded['g'] * (1. + gamma) + beta


def _norm(x, n):
    return layer_norm(x) * n['scale'] + n['bias']


def macro_logits(params: dict, encoded: dict) -> jax.Array:
    heads = params['heads']
    return network.dense(_norm(encoded['plan'], heads['macro_norm']), heads['macro'])


def value_heads(params: dict, encoded: dict) -> dict:
    """WDL, score-margin and turns-remaining logits from the VALUE token, and per-tile ownership in [-1, 1]."""
    heads = params['heads']
    v = encoded['value']
    hidden = jax.nn.silu(network.dense(_norm(v, heads['wdl_norm']), heads['wdl_hidden']))
    score_hidden = jax.nn.silu(network.dense(v, heads['score_hidden']))
    return dict(wdl=network.dense(hidden, heads['wdl']),
                score=network.dense(score_hidden, heads['score']),
                turns=network.dense(v, heads['turns']),
                ownership=jnp.tanh(network.dense(encoded['tiles'], heads['ownership'])[:, 0]),
                **belief_heads(params, encoded))


def belief_heads(params: dict, encoded: dict) -> dict:
    """The opponent's hidden state from the VALUE token (own view only): tech logits [TECH_SLOTS] (one yes/no per tech)
    and economy logits [len(ECONOMY), ECONOMY_BUCKETS] (stars, income, cities, total city level, units)."""
    heads = params['heads']
    hidden = jax.nn.silu(network.dense(_norm(encoded['value'], heads['belief_norm']), heads['belief_hidden']))
    return dict(belief_tech=network.dense(hidden, heads['belief_tech']),
                belief_economy=network.dense(hidden, heads['belief_economy']).reshape(len(ECONOMY), ECONOMY_BUCKETS))


def economy_buckets(stars, income, cities, city_levels, units):
    """[..., len(ECONOMY)] bucket labels of the opponent's economy (ECONOMY order)."""
    values = dict(stars=stars, income=income, cities=cities, city_levels=city_levels, units=units)
    return jnp.stack([jnp.clip(jnp.asarray(values[k]) // ECONOMY_SCALE[k], 0, ECONOMY_BUCKETS - 1) for k in ECONOMY],
                     -1).astype(jnp.int32)


def tech_mask() -> np.ndarray:
    """[TECH_SLOTS] True for the real technology ids of the rules (gpu/rules.json)."""
    rules = json.loads((Path(__file__).parents[1] / 'gpu' / 'rules.json').read_text())
    mask = np.zeros(TECH_SLOTS, bool)
    for t in rules['technologies']:
        if 0 <= t['id'] < TECH_SLOTS:
            mask[t['id']] = True
    return mask


def wdl_value(logits: jax.Array) -> jax.Array:
    """Scalar value in [-1, 1] of WDL logits: p(win) - p(loss)."""
    p = jax.nn.softmax(logits, axis=-1)
    return p[..., WIN] - p[..., LOSS]


def critic_wdl(params: dict, critic: dict, tokens: jax.Array, opp_board: jax.Array, opp_glob: jax.Array) -> jax.Array:
    """Privileged critic WDL logits: transformer_v3's critic branch on the 50 public tokens and the other seat's view."""
    policy = params['policy']
    size = opp_board.shape[0]
    patch = transformer_v3._patch_of(policy)
    grid = transformer_v3._grid(size, patch)
    opp_board = fit_board(opp_board, critic['priv_stem']['w'].shape[0])
    opp_glob = fit_global(opp_glob, policy['global_token']['w'].shape[0])
    priv, _ = transformer_v3._stem(critic['priv_stem'], critic.get('priv_patch'), opp_board, size, patch)
    x = tokens + jnp.concatenate((priv, linear(opp_glob, critic['priv_global'])[None]), axis=0)
    x = layer_norm(transformer_v3._run_blocks(x, critic['blocks'], grid))
    pooled = jnp.concatenate((x[:-1].mean(0), x[:-1].max(0), x[-1]))
    hidden = jax.nn.silu(layer_norm(linear(pooled, critic['hidden'])))
    return linear(hidden, critic['wdl'])


# ---------------------------------------------------------------- acting and evaluating

def act(params: dict, features, masks, key, plan, *, size: int, deterministic: bool = False) -> tuple:
    """Play from the own view: (choice, path masks, logp, outputs). outputs has the macro logits (for choosing the
    next plan), the value heads and the scalar value p(win) - p(loss) for search."""
    encoded = encode(params, features.board, features.global_state, plan)
    conditioned, tiles, g = _conditioned(params, encoded)
    choice, path, logp = network.sample_from(conditioned, tiles, g, masks, key, size=size, deterministic=deterministic)
    heads = value_heads(params, encoded)
    return choice, path, logp, dict(macro=macro_logits(params, encoded), value=wdl_value(heads['wdl']), **heads)


def choose_plan(params: dict, features, previous_plan, key, *, deterministic: bool = False):
    """The plan for a new turn: sampled from the macro head of a pass conditioned on the previous plan. Returns
    (plan, its log-probability)."""
    encoded = encode(params, features.board, features.global_state, previous_plan)
    logp = jax.nn.log_softmax(macro_logits(params, encoded))
    plan = jnp.argmax(logp) if deterministic else jax.random.categorical(key, logp)
    return plan.astype(jnp.int32), logp[plan]


def evaluate(params: dict, critic: dict, features, choice, masks, plan) -> dict:
    """The learner's view of one row (features = own view then the other seat's view, as transformer_v3): action
    log-probability and entropy, macro logits, value heads and critic WDL logits, from ONE public encoder pass."""
    own_board, own_glob, opp_board, opp_glob = transformer_v3._split_views(features)
    encoded = encode(params, own_board, own_glob, plan)
    logp, entropy = _action_logp(params, encoded, choice, masks)
    return dict(logp=logp, entropy=entropy, macro=macro_logits(params, encoded),
                critic_wdl=critic_wdl(params, critic, encoded['tokens'], opp_board, opp_glob),
                **value_heads(params, encoded))


def _action_logp(params: dict, encoded: dict, choice, masks) -> tuple:
    """(log-probability, entropy) of a stored action path under the plan-conditioned policy."""
    conditioned, tiles, g = _conditioned(params, encoded)
    logits = (network.dense(g, conditioned['family']), network.dense(tiles, conditioned['source'])[:, choice.family],
              network.target_logits(conditioned, tiles, g, choice.family, choice.source))
    pairs = [network.logprob_entropy(l, m, a) for l, m, a in zip(logits, masks, choice, strict=True)]
    return sum(p[0] for p in pairs), sum(p[1] for p in pairs)


def action_logp(params: dict, board, global_state, choice, masks, plan) -> jax.Array:
    """log pi(action | own view, plan) for one row, from the own view only (no critic)."""
    return _action_logp(params, encode(params, board, global_state, plan), choice, masks)[0]


def family_probs(params: dict, board, global_state, family_mask, plan) -> jax.Array:
    """The plan-conditioned policy's distribution over action families (legal ones only)."""
    conditioned, _, g = _conditioned(params, encode(params, board, global_state, plan))
    logits = network.dense(g, conditioned['family'])
    return jax.nn.softmax(jnp.where(family_mask, logits, -1e9))


def plan_loss(params: dict, rows: dict, starts: dict, *, macro_coef=1.) -> tuple:
    """Stage B: hindsight plan imitation on self-play.

      rows: board, global_state, choice, masks, plan [rows] = the label of the turn the action belongs to.
        Loss: -log pi(action | state, label), so each plan learns to play like its cluster of turns.
      starts: board, global_state, previous [rows] (the seat's previous label, 0 at game start), label [rows], at the
        first decision of a turn. Loss: macro cross-entropy, read from a pass conditioned on `previous` as choose_plan.

    Differentiate with respect to the PLAN_HEADS group only (value heads and Plato frozen)."""
    logp = jax.vmap(action_logp, in_axes=(None, 0, 0, 0, 0, 0))(
        params, rows['board'], rows['global_state'], rows['choice'], rows['masks'], rows['plan'])
    macro = jax.vmap(lambda b, g, z: macro_logits(params, encode(params, b, g, z)))(
        starts['board'], starts['global_state'], starts['previous'])
    macro_ce = _cross_entropy(macro, starts['label']).mean()
    bc = -logp.mean()
    total = bc + macro_coef * macro_ce
    return total, dict(loss=total, bc=bc, macro=macro_ce,
                       macro_accuracy=jnp.mean(jnp.argmax(macro, -1) == starts['label']))


# ---------------------------------------------------------------- targets and losses

def outcome_class(reward, capped):
    """WDL class of a finished game for one seat. A game that reached the round cap is a draw here, even though PPO
    still scores it -1 for both seats (truncation_reward)."""
    reward, capped = jnp.asarray(reward), jnp.asarray(capped, bool)
    return jnp.where(capped, DRAW, jnp.where(reward > 0, WIN, LOSS)).astype(jnp.int32)


def score_bucket(margin):
    """Bucket of a final score margin (own minus opponent)."""
    half = SCORE_BUCKETS // 2
    return (jnp.clip(jnp.round(jnp.asarray(margin) / SCORE_SCALE), -half, half) + half).astype(jnp.int32)


def turns_bucket(turns_remaining):
    return jnp.clip(jnp.asarray(turns_remaining) // TURN_SCALE, 0, TURN_BUCKETS - 1).astype(jnp.int32)


def _cross_entropy(logits, label):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), label[..., None], axis=-1)[..., 0]


def head_loss(params: dict, critic: dict, batch: dict, *, wdl_coef=1., distill_coef=1., critic_coef=1., score_coef=.1,
              turns_coef=.1, ownership_coef=.1, macro_coef=1., tech_coef=.1, economy_coef=.1) -> tuple:
    """Supervised losses for the new heads, over a batch of rows:

      features, choice, masks (as ppo), plan [rows]: the plan in force
      outcome [rows] (outcome_class), score [rows] (score_bucket), turns [rows] (turns_bucket),
      ownership [rows, tiles] in {-1, 0, 1} (final owner: own, nobody, opponent),
      plan_label [rows]: a plan to imitate (e.g. a clustered next-turns statistic), or -1 for none.
      optional opponent state: opp_tech [rows, TECH_SLOTS] bool, economy [rows, len(ECONOMY)] (economy_buckets),
        has_belief [rows] (False for rows without these labels). Tech: binary cross-entropy over the real tech ids;
        economy: mean cross-entropy over the five targets.

    WDL head: cross-entropy to the outcome plus KL from the critic's (stop-gradient) WDL. The critic: cross-entropy to
    the outcome. To train only the new heads (frozen trunk), differentiate with respect to params['heads'] only."""
    out = jax.vmap(evaluate, in_axes=(None, None, 0, 0, 0, 0))(params, critic, batch['features'], batch['choice'],
                                                                 batch['masks'], batch['plan'])
    teacher = jax.nn.softmax(jax.lax.stop_gradient(out['critic_wdl']))
    student = jax.nn.log_softmax(out['wdl'])
    distill = jnp.sum(teacher * (jnp.log(teacher + 1e-12) - student), axis=-1).mean()
    labelled = batch['plan_label'] >= 0
    macro = (jnp.where(labelled, _cross_entropy(out['macro'], jnp.maximum(batch['plan_label'], 0)), 0.).sum()
             / jnp.maximum(labelled.sum(), 1))
    terms = dict(wdl=_cross_entropy(out['wdl'], batch['outcome']).mean(), distill=distill,
                 critic=_cross_entropy(out['critic_wdl'], batch['outcome']).mean(),
                 score=_cross_entropy(out['score'], batch['score']).mean(),
                 turns=_cross_entropy(out['turns'], batch['turns']).mean(),
                 ownership=jnp.mean((out['ownership'] - batch['ownership']) ** 2), macro=macro)
    coefs = dict(wdl=wdl_coef, distill=distill_coef, critic=critic_coef, score=score_coef, turns=turns_coef,
                 ownership=ownership_coef, macro=macro_coef)
    extra = {}
    if 'opp_tech' in batch:
        has = batch['has_belief'].astype(jnp.float32)
        rows = jnp.maximum(has.sum(), 1.)
        valid = jnp.asarray(tech_mask())
        target = batch['opp_tech'].astype(jnp.float32)
        logits = out['belief_tech']
        bce = jnp.maximum(logits, 0) - logits * target + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        per_row = (bce * valid).sum(-1) / valid.sum()
        terms['tech'] = (per_row * has).sum() / rows
        right = ((logits > 0) == (target > .5)) & valid
        extra['tech_accuracy'] = ((right.sum(-1) / valid.sum()) * has).sum() / rows
        economy_ce = _cross_entropy(out['belief_economy'], batch['economy'])          # [rows, len(ECONOMY)]
        terms['economy'] = (economy_ce.mean(-1) * has).sum() / rows
        hit = (jnp.argmax(out['belief_economy'], -1) == batch['economy']) * has[:, None]
        extra.update({f'{k}_accuracy': hit[:, i].sum() / rows for i, k in enumerate(ECONOMY)})
        coefs.update(tech=tech_coef, economy=economy_coef)
    total = sum(coefs[k] * v for k, v in terms.items())
    accuracy = jnp.mean(jnp.argmax(out['wdl'], -1) == batch['outcome'])
    return total, dict(loss=total, wdl_accuracy=accuracy, **terms, **extra)


# ---------------------------------------------------------------- checkpoints

def save(path, params: dict, critic: dict, spec: dict, **meta) -> None:
    """`path` (.npz, flattened params), `<stem>-critic.npz` and `<stem>.json` (spec and counts). spec is the base
    network's (size, width, blocks, architecture, obs_version) as distributed.network_spec reports it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for tree, target in ((params, path), (critic, path.with_name(path.stem + '-critic.npz'))):
        np.savez(target, **{f'p{i}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
    info = dict(architecture=ARCHITECTURE, plan_count=PLAN_COUNT, base=spec, parameters=parameter_counts(params, critic),
                **meta)
    path.with_suffix('.json').write_text(json.dumps(info, indent=1) + '\n')


def template(spec: dict) -> tuple:
    """Shape-only (params, critic) for a base spec, for load()."""
    from .distributed import feature_shapes
    board, glob = feature_shapes(spec.get('players', 2), spec.get('obs_version', 1))
    return jax.eval_shape(lambda: init(jax.random.key(0), board, glob, size=spec['size'], width=spec['width'],
                                       blocks=spec['blocks'], architecture=spec['architecture']))


# Heads added after the first heroX checkpoints, newest last. An older checkpoint loads without them; they then take
# init_heads' values (head FiLM and the family bias zero, the belief heads random as in a fresh conversion, seed 0).
LATER_HEADS = (('head_film', 'family_bias'), BELIEF_HEADS)


def load(path) -> tuple:
    path = Path(path)
    info = json.loads(path.with_suffix('.json').read_text())
    if info.get('architecture') != ARCHITECTURE:
        raise ValueError(f'not a {ARCHITECTURE} checkpoint')
    trees = template(info['base'])
    out = []
    for tree, source in zip(trees, (path, path.with_name(path.stem + '-critic.npz'))):
        with np.load(source, allow_pickle=False) as archive:
            stored = len(archive.files)
            missing = ()
            if 'heads' in tree and stored != len(jax.tree.leaves(tree)):
                for newest in range(1, len(LATER_HEADS) + 1):
                    missing = sum(LATER_HEADS[-newest:], ())
                    if stored == len(jax.tree.leaves({k: v for k, v in tree['heads'].items() if k not in missing})) + \
                            len(jax.tree.leaves({k: v for k, v in tree.items() if k != 'heads'})):
                        break
                tree = dict(tree, heads={k: v for k, v in tree['heads'].items() if k not in missing})
            leaves, structure = jax.tree.flatten(tree)
            values = [archive[f'p{i}'] for i in range(len(leaves))]
        if any(v.shape != l.shape for v, l in zip(values, leaves, strict=True)):
            raise ValueError(f'{source}: tensor shapes do not match the heroX template')
        loaded = structure.unflatten([jnp.asarray(v) for v in values])
        if missing:
            policy = loaded['policy']
            fresh = init_heads(jax.random.key(0), policy['global_token']['w'].shape[1], policy['family']['w'].shape[1])
            loaded = dict(loaded, heads={**loaded['heads'], **{k: fresh[k] for k in missing}})
        out.append(loaded)
    return tuple(out)
