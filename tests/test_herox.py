"""heroX (rl/herox.py): parameter counts, exact Plato equivalence at conversion, the readout mask, plan conditioning,
the information contract, the losses and the checkpoint round trip."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from polytopia_bot.rl import herox, network, transformer_v3
from polytopia_bot.rl.actions import Choice
from polytopia_bot.rl.distributed import _critic_template, _template_params, feature_shapes
from polytopia_bot.rl.observation import OBS_VERSION, Features

from test_transformer_v3 import SIZE, TILES, _engine_masks, _features, _full, _masks

ROOT = Path(__file__).resolve().parents[1]
PLATO = ROOT / 'weights' / 'plato' / 'final.npz'


def _spec(architecture='transformer-v3-p2-h4', width=32, blocks=2):
    return dict(size=SIZE, blocks=blocks, width=width, architecture=architecture, obs_version=OBS_VERSION)


@pytest.fixture(scope='module')
def shapes():
    return feature_shapes(2, OBS_VERSION)


def _base(spec):
    # nonzero relative biases and slightly moved weights, so equivalence is not trivially about zeros
    policy = jax.tree.map(lambda a: a + .01, _template_params(**spec))
    return policy, _critic_template(**spec)


def _trained(params, key=jax.random.key(42)):
    """The same network with FiLM and the query gate moved off zero, as training would."""
    heads = params['heads']
    heads = dict(heads, plan_query_gate=jnp.float32(.5),
                 film=dict(heads['film'], w=jax.random.normal(key, heads['film']['w'].shape) * .05))
    return dict(params, heads=heads)


def test_parameter_counts_match_the_design():
    """Width 128, 6 blocks, -p2-h4 at 14x14 (Plato's shape): 196,766 new parameters on Plato's 3,212,922."""
    params, critic = herox.template(_spec(width=128, blocks=6))
    counts = herox.parameter_counts(params, critic)
    assert counts == dict(base_policy=3_212_922, new_heads=196_766, policy=3_409_688, critic=401_063)


def test_conversion_plays_exactly_as_the_base_network(shapes):
    spec = _spec()
    policy, critic = _base(spec)
    params, wdl_critic = herox.from_plato(policy, critic)
    own, opp = _features(jax.random.key(1), *shapes), _features(jax.random.key(2), *shapes)
    masks, key = _engine_masks(jax.random.key(3)), jax.random.key(4)
    want = network.sample(policy, own, masks, key, size=SIZE)
    for plan in (0, 7, herox.PLAN_COUNT - 1):
        choice, path, logp, _ = herox.act(params, own, masks, key, plan, size=SIZE)
        assert tuple(int(c) for c in choice) == tuple(int(c) for c in want[0])
        np.testing.assert_allclose(logp, want[2], rtol=1e-5, atol=1e-5)
    # the evaluated log-probability and entropy too, and the critic keeps the base critic's value as win - loss logits
    full, choice = _full(own, opp), Choice(jnp.int32(0), jnp.int32(3), jnp.int32(5))
    path = _masks(jax.random.key(5))
    logp, base_value, entropy = transformer_v3.evaluate(policy, critic, full, choice, path)
    out = herox.evaluate(params, wdl_critic, full, choice, path, 3)
    np.testing.assert_allclose(out['logp'], logp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out['entropy'], entropy, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out['critic_wdl'][herox.WIN] - out['critic_wdl'][herox.LOSS], 2 * base_value,
                               rtol=1e-4, atol=1e-5)


def test_readout_tokens_never_reach_the_public_tokens(shapes):
    """The PLAN and VALUE tokens can be anything: the policy does not change (the -1e9 mask), but their outputs do."""
    params, _ = herox.from_plato(*_base(_spec()))
    own = _features(jax.random.key(6), *shapes)
    a = herox.encode(params, own.board, own.global_state, 0)
    noise = jax.random.normal(jax.random.key(60), (2, params['heads']['plan_token'].shape[0]))
    moved = dict(params, heads=dict(params['heads'], plan_token=params['heads']['plan_token'] + noise[0],
                                    value_token=params['heads']['value_token'] + noise[1]))
    b = herox.encode(moved, own.board, own.global_state, 0)
    for name in ('tiles', 'g', 'tokens'):
        np.testing.assert_array_equal(a[name], b[name])
    assert float(jnp.abs(a['value'] - b['value']).max()) > 1e-3
    assert float(jnp.abs(a['plan'] - b['plan']).max()) > 1e-3


def test_attention_bias_layout():
    grid, heads = 7, 4
    table = jnp.arange(heads * 13 * 13, dtype=jnp.float32).reshape(heads, 13, 13)
    bias = herox.attention_bias(table, grid)
    public = grid * grid + 1
    assert bias.shape == (heads, public + 2, public + 2)
    # the patch block is Plato's relative bias, GLOBAL is unbiased, readout keys are masked for public queries
    from polytopia_bot.rl.transformer import relative_attention_bias
    np.testing.assert_array_equal(bias[:, :public, :public], relative_attention_bias(table, grid, grid))
    assert bool(jnp.all(bias[:, :public, public:] == herox.MASKED))
    assert bool(jnp.all(bias[:, public:, :] == 0.))


def test_plan_conditions_the_policy_once_trained(shapes):
    params = _trained(herox.from_plato(*_base(_spec()))[0])
    own = _features(jax.random.key(7), *shapes)
    choice, path = Choice(jnp.int32(0), jnp.int32(3), jnp.int32(5)), _masks(jax.random.key(8))
    full = _full(own, _features(jax.random.key(9), *shapes))
    critic = herox.critic_from_v3(_critic_template(**_spec()))
    a = herox.evaluate(params, critic, full, choice, path, 1)
    b = herox.evaluate(params, critic, full, choice, path, 2)
    assert abs(float(a['logp'] - b['logp'])) > 1e-4
    assert float(jnp.abs(a['macro'] - b['macro']).max()) > 1e-4


def test_public_heads_never_read_privileged_input(shapes):
    """Vary only the other seat's view: the policy, macro and every value-token head are unchanged exactly, their
    gradients w.r.t. the privileged input are zero, and the critic responds."""
    params = _trained(herox.from_plato(*_base(_spec()))[0])
    critic = herox.critic_from_v3(jax.tree.map(lambda a: a + .05, _critic_template(**_spec())))
    own = _features(jax.random.key(10), *shapes)
    opp_a, opp_b = _features(jax.random.key(11), *shapes), _features(jax.random.key(12), *shapes)
    choice, path = Choice(jnp.int32(0), jnp.int32(3), jnp.int32(5)), _masks(jax.random.key(13))
    a = herox.evaluate(params, critic, _full(own, opp_a), choice, path, 4)
    b = herox.evaluate(params, critic, _full(own, opp_b), choice, path, 4)
    for name in ('logp', 'entropy', 'macro', 'wdl', 'score', 'turns', 'ownership'):
        np.testing.assert_array_equal(a[name], b[name])
    assert float(jnp.abs(a['critic_wdl'] - b['critic_wdl']).max()) > 1e-6

    def public(opp_board, opp_glob):
        out = herox.evaluate(params, critic, _full(own, Features(opp_board, opp_glob)), choice, path, 4)
        return out['logp'] + out['macro'].sum() + out['wdl'].sum() + out['score'].sum() + out['ownership'].sum()
    grads = jax.grad(public, argnums=(0, 1))(opp_a.board, opp_a.global_state)
    assert all(float(jnp.abs(g).max()) == 0. for g in grads)


def _batch(shapes, rows=4):
    feats = jax.vmap(lambda k: _full(_features(k, *shapes), _features(jax.random.fold_in(k, 1), *shapes)))(
        jax.random.split(jax.random.key(14), rows))
    return dict(features=feats, masks=jax.vmap(_masks)(jax.random.split(jax.random.key(15), rows)),
                choice=Choice(jnp.zeros(rows, jnp.int32), jnp.full(rows, 3, jnp.int32), jnp.full(rows, 5, jnp.int32)),
                plan=jnp.array([0, 1, 2, 3]),
                outcome=herox.outcome_class(jnp.array([1., -1., -1., 1.]), jnp.array([False, False, True, False])),
                score=herox.score_bucket(jnp.array([2500., -400., 0., 30000.])),
                turns=herox.turns_bucket(jnp.array([0, 15, 300, 42])),
                ownership=jax.random.randint(jax.random.key(16), (rows, TILES), -1, 2).astype(jnp.float32),
                plan_label=jnp.array([5, -1, 2, -1]))


def test_targets():
    np.testing.assert_array_equal(herox.outcome_class(jnp.array([1., -1., -1., 1.]), jnp.array([0, 0, 1, 1])),
                                  [herox.WIN, herox.LOSS, herox.DRAW, herox.DRAW])
    np.testing.assert_array_equal(herox.score_bucket(jnp.array([0., 2500., -400., 1e6, -1e6])), [10, 12, 10, 20, 0])
    np.testing.assert_array_equal(herox.turns_bucket(jnp.array([0, 9, 10, 199, 500])), [0, 0, 1, 19, 19])


def test_head_loss_trains_the_new_heads(shapes):
    params, critic = herox.from_plato(*_base(_spec()))
    batch = _batch(shapes)
    (total, metrics), grads = jax.value_and_grad(herox.head_loss, has_aux=True)(params, critic, batch)
    assert np.isfinite(float(total)) and all(np.isfinite(float(v)) for v in metrics.values())
    heads = grads['heads']
    for name in ('value_token', 'plan_token', 'wdl', 'score', 'turns', 'ownership', 'macro', 'film'):
        assert float(max(jnp.abs(x).max() for x in jax.tree.leaves(heads[name]))) > 0., name
    # at conversion the plan embedding is reached only through zero-initialized layers (FiLM, the query gate); it gets
    # a gradient once FiLM has moved, and the gate gets one from the policy loss (PPO)
    assert float(jnp.abs(heads['plan_embedding']).max()) == 0.
    moved = _trained(params)
    later = jax.grad(lambda p: herox.head_loss(p, critic, batch)[0])(moved)['heads']
    assert float(jnp.abs(later['plan_embedding']).max()) > 0.
    gate = jax.grad(lambda p: herox.evaluate(p, critic, jax.tree.map(lambda a: a[0], batch['features']),
                                             jax.tree.map(lambda a: a[0], batch['choice']),
                                             jax.tree.map(lambda a: a[0], batch['masks']), 1)['logp'])(params)
    assert float(jnp.abs(gate['heads']['plan_query_gate'])) > 0.
    # the critic is trained by its own cross-entropy; the distillation term never moves it
    critic_grads = jax.grad(lambda c: herox.head_loss(params, c, batch, critic_coef=0.)[0])(critic)
    assert all(float(jnp.abs(x).max()) == 0. for x in jax.tree.leaves(critic_grads))


def test_frozen_trunk_training_leaves_plato_untouched(shapes):
    """Differentiating with respect to the heads alone gives the step-1 recipe: one optimizer step on the heads leaves
    the base policy bit-identical."""
    params, critic = herox.from_plato(*_base(_spec()))
    batch = _batch(shapes)
    grads = jax.grad(lambda h: herox.head_loss(dict(params, heads=h), critic, batch)[0])(params['heads'])
    updated = dict(params, heads=jax.tree.map(lambda p, g: p - 1e-2 * g, params['heads'], grads))
    for a, b in zip(jax.tree.leaves(updated['policy']), jax.tree.leaves(params['policy'])):
        assert bool(jnp.all(a == b))


def test_choose_plan(shapes):
    params = _trained(herox.from_plato(*_base(_spec()))[0])
    own = _features(jax.random.key(17), *shapes)
    plan, logp = herox.choose_plan(params, own, 0, jax.random.key(18))
    assert 0 <= int(plan) < herox.PLAN_COUNT and float(logp) <= 0.
    greedy, _ = herox.choose_plan(params, own, 0, jax.random.key(19), deterministic=True)
    encoded = herox.encode(params, own.board, own.global_state, 0)
    assert int(greedy) == int(jnp.argmax(herox.macro_logits(params, encoded)))


def test_checkpoint_round_trip(tmp_path):
    spec = _spec()
    params, critic = herox.from_plato(*_base(spec))
    params = _trained(params)
    path = tmp_path / 'herox.npz'
    herox.save(path, params, critic, spec, source='test')
    loaded_params, loaded_critic = herox.load(path)
    for want, got in ((params, loaded_params), (critic, loaded_critic)):
        assert all(bool(jnp.all(a == b)) for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True))


@pytest.mark.skipif(not PLATO.exists(), reason='weights/plato is not present')
def test_real_plato_converts_exactly(shapes):
    """On the released Plato weights: heroX made from them chooses Plato's moves with Plato's probabilities."""
    from polytopia_bot.rl.distributed import network_spec
    from polytopia_bot.rl.runner import load_checkpoint
    spec = network_spec(PLATO)
    policy = load_checkpoint(PLATO, _template_params(**spec))
    critic = load_checkpoint(PLATO.with_name('final-critic.npz'), _critic_template(**spec))
    params, wdl_critic = herox.from_plato(policy, critic)
    assert herox.parameter_counts(params, wdl_critic)['policy'] == 3_409_688
    for i in range(3):
        own = _features(jax.random.key(100 + i), *shapes)
        masks, key = _engine_masks(jax.random.key(200 + i)), jax.random.key(300 + i)
        want = network.sample(policy, own, masks, key, size=SIZE)
        got = herox.act(params, own, masks, key, i, size=SIZE)
        assert tuple(int(c) for c in got[0]) == tuple(int(c) for c in want[0])
        np.testing.assert_allclose(got[2], want[2], rtol=1e-4, atol=1e-4)


def _plan_rows(shapes):
    batch = _batch(shapes)
    own_board, own_glob, _, _ = jax.vmap(transformer_v3._split_views)(batch['features'])
    rows = dict(board=own_board, global_state=own_glob, choice=batch['choice'], masks=batch['masks'], plan=batch['plan'])
    starts = dict(board=own_board, global_state=own_glob, previous=jnp.array([0, 3, 3, 7]), label=jnp.array([3, 3, 7, 1]))
    return rows, starts


def test_plan_loss_trains_only_the_plan_group(shapes):
    """Stage B: gradients reach the PLAN_HEADS (FiLM through the policy, the macro head through its cross-entropy) and
    never the value heads; the value stage's group and the plan group partition the heads."""
    params, _ = herox.from_plato(*_base(_spec()))
    assert set(herox.VALUE_HEADS) | set(herox.PLAN_HEADS) == set(params['heads'])
    assert not set(herox.VALUE_HEADS) & set(herox.PLAN_HEADS)
    rows, starts = _plan_rows(shapes)
    (total, metrics), grads = jax.value_and_grad(herox.plan_loss, has_aux=True)(params, rows, starts)
    assert np.isfinite(float(total)) and all(np.isfinite(float(v)) for v in metrics.values())
    for name in ('film', 'plan_query_gate', 'macro', 'plan_token', 'family_bias', 'head_film'):
        assert float(max(jnp.abs(x).max() for x in jax.tree.leaves(grads['heads'][name]))) > 0., name
    # value_token is the one exception: the PLAN token attends to it (readouts see all 52 tokens); stage B freezes it
    for name in set(herox.VALUE_HEADS) - {'value_token'}:
        assert all(float(jnp.abs(x).max()) == 0. for x in jax.tree.leaves(grads['heads'][name])), name
    # at conversion every plan gives Plato's log-probability, so the imitation term is Plato's NLL
    plato = jax.vmap(lambda b, g, c, m: herox.action_logp(params, b, g, c, m, 0))(
        rows['board'], rows['global_state'], rows['choice'], rows['masks'])
    np.testing.assert_allclose(float(metrics['bc']), -float(plato.mean()), rtol=1e-5)


def test_family_probs_sum_to_one_over_legal_families(shapes):
    params, _ = herox.from_plato(*_base(_spec()))
    rows, _ = _plan_rows(shapes)
    p = herox.family_probs(params, rows['board'][0], rows['global_state'][0], rows['masks'].family[0], 2)
    assert abs(float(p.sum()) - 1.) < 1e-5
    assert float(jnp.where(rows['masks'].family[0], 0., p).sum()) < 1e-6


def test_head_side_plan_moves_the_policy_but_not_the_value(shapes):
    """The stage-B parameters (head FiLM, family bias) change the action distribution per plan while every token, so
    the value heads and the macro head's input, stays exactly as before."""
    params, _ = herox.from_plato(*_base(_spec()))
    rows, _ = _plan_rows(shapes)
    heads = params['heads']
    moved = dict(params, heads=dict(heads, family_bias=jax.random.normal(jax.random.key(3), heads['family_bias'].shape),
                                    head_film=dict(heads['head_film'], w=jax.random.normal(
                                        jax.random.key(4), heads['head_film']['w'].shape) * .1)))
    board, glob, choice, masks = rows['board'][0], rows['global_state'][0], jax.tree.map(lambda a: a[0], rows['choice']), \
        jax.tree.map(lambda a: a[0], rows['masks'])
    for z in (0, 5):
        before, after = herox.encode(params, board, glob, z), herox.encode(moved, board, glob, z)
        np.testing.assert_array_equal(before['value'], after['value'])
        np.testing.assert_array_equal(herox.value_heads(params, before)['wdl'], herox.value_heads(moved, after)['wdl'])
    a, b = (herox.action_logp(moved, board, glob, choice, masks, z) for z in (0, 5))
    assert abs(float(a) - float(b)) > 1e-3
    assert abs(float(herox.action_logp(params, board, glob, choice, masks, 0))
               - float(herox.action_logp(params, board, glob, choice, masks, 5))) < 1e-6


def test_checkpoints_without_head_side_parameters_still_load(tmp_path):
    params, critic = herox.from_plato(*_base(_spec()))
    head_side, belief = herox.LATER_HEADS
    for missing in (belief, head_side + belief):
        old = dict(params, heads={k: v for k, v in params['heads'].items() if k not in missing})
        herox.save(tmp_path / 'old.npz', old, critic, _spec())
        loaded, _ = herox.load(tmp_path / 'old.npz')
        assert set(loaded['heads']) == set(params['heads'])
        for k in set(missing) & set(head_side):     # zero: the policy stays Plato's
            assert all(float(jnp.abs(x).max()) == 0. for x in jax.tree.leaves(loaded['heads'][k]))
        # the belief heads are initialized as in a fresh conversion (random, so they can train)
        assert float(jnp.abs(loaded['heads']['belief_hidden']['w']).max()) > 0.
        np.testing.assert_array_equal(loaded['heads']['plan_embedding'], params['heads']['plan_embedding'])


def test_search_scores_lanes_with_the_value_token(shapes):
    """search._value: a heroX checkpoint as value_params scores with p(win) - p(loss) of its value token."""
    from polytopia_bot.rl import search
    params, _ = herox.from_plato(*_base(_spec()))
    rows, _ = _plan_rows(shapes)
    features = Features(rows['board'][0], rows['global_state'][0])
    expected = herox.wdl_value(herox.value_heads(params, herox.encode(params, features.board, features.global_state, 0))['wdl'])
    np.testing.assert_allclose(float(search._value(params, features)), float(expected), rtol=1e-6)
    assert -1. <= float(expected) <= 1.


def _belief_batch(shapes):
    batch = _batch(shapes)
    rows = len(batch['plan'])
    tech = jax.random.bernoulli(jax.random.key(21), .4, (rows, herox.TECH_SLOTS))
    economy = herox.economy_buckets(jnp.array([3, 40, 0, 200]), jnp.array([2, 9, 0, 50]), jnp.array([1, 4, 0, 30]),
                                    jnp.array([1, 9, 0, 99]), jnp.array([2, 11, 0, 90]))
    return dict(batch, opp_tech=tech, economy=economy, has_belief=jnp.array([True, True, False, True]))


def test_economy_buckets_and_tech_mask():
    b = herox.economy_buckets(jnp.array([0, 4, 5, 999]), jnp.array([0, 1, 2, 999]), jnp.array([0, 1, 2, 999]),
                              jnp.array([0, 2, 3, 999]), jnp.array([0, 1, 2, 999]))
    np.testing.assert_array_equal(b[:, 0], [0, 0, 1, herox.ECONOMY_BUCKETS - 1])          # stars, width 5
    np.testing.assert_array_equal(b[:, 2], [0, 1, 2, herox.ECONOMY_BUCKETS - 1])          # cities, width 1
    assert b.shape == (4, len(herox.ECONOMY))
    mask = herox.tech_mask()
    assert mask.shape == (herox.TECH_SLOTS,) and 30 <= mask.sum() < herox.TECH_SLOTS


def test_belief_heads_train_from_the_own_view_only(shapes):
    """The belief losses reach the belief heads and the value token, never Plato's weights through a heads-only
    gradient; rows without labels (has_belief False) contribute nothing; the heads never read the opponent's view."""
    params, critic = herox.from_plato(*_base(_spec()))
    batch = _belief_batch(shapes)
    (total, metrics), grads = jax.value_and_grad(lambda h: herox.head_loss(dict(params, heads=h), critic, batch),
                                                 has_aux=True)(params['heads'])
    for k in ('tech', 'economy', 'tech_accuracy', 'stars_accuracy', 'units_accuracy'):
        assert np.isfinite(float(metrics[k])), k
    for name in herox.BELIEF_HEADS:
        assert float(max(jnp.abs(x).max() for x in jax.tree.leaves(grads[name]))) > 0., name
    # unlabelled rows: changing their targets changes nothing
    other = dict(batch, opp_tech=batch['opp_tech'].at[2].set(~batch['opp_tech'][2]), economy=batch['economy'].at[2].set(7))
    _, m2 = herox.head_loss(params, critic, other)
    np.testing.assert_allclose(float(metrics['tech']), float(m2['tech']), rtol=1e-6)
    np.testing.assert_allclose(float(metrics['economy']), float(m2['economy']), rtol=1e-6)
    # the heads read the own view only: the opponent's half of the features does not move them
    own_board, own_glob, opp_board, opp_glob = transformer_v3._split_views(jax.tree.map(lambda a: a[0], batch['features']))
    a = herox.belief_heads(params, herox.encode(params, own_board, own_glob, 0))
    assert a['belief_tech'].shape == (herox.TECH_SLOTS,) and a['belief_economy'].shape == (len(herox.ECONOMY),
                                                                                           herox.ECONOMY_BUCKETS)
