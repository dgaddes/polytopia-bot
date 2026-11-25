"""heroX checkpoints: convert Plato into heroX and inspect a heroX checkpoint.

    .venv/bin/python scripts/herox.py convert --out runs/herox/init.npz
    .venv/bin/python scripts/herox.py info runs/herox/init.npz
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def convert(args):
    import jax
    from polytopia_bot.rl import herox
    from polytopia_bot.rl.distributed import _critic_template, _template_params, network_spec
    from polytopia_bot.rl.runner import load_checkpoint, sha256
    source = Path(args.plato)
    spec = network_spec(source)
    if not spec['architecture'].startswith('transformer-v3'):
        raise SystemExit(f'{source} is {spec["architecture"]}; heroX needs a transformer-v3 checkpoint')
    critic_path = source.with_name(source.stem + '-critic.npz')
    policy = load_checkpoint(source, _template_params(**spec))
    critic = load_checkpoint(critic_path, _critic_template(**spec))
    params, wdl_critic = herox.from_plato(policy, critic, jax.random.key(args.seed))
    herox.save(args.out, params, wdl_critic, spec, source=str(source), source_sha256=sha256(source),
               source_critic_sha256=sha256(critic_path), seed=args.seed)
    print(json.dumps(dict(out=args.out, base=spec, parameters=herox.parameter_counts(params, wdl_critic)), indent=1))


def info(args):
    from polytopia_bot.rl import herox
    params, critic = herox.load(args.checkpoint)
    meta = json.loads(Path(args.checkpoint).with_suffix('.json').read_text())
    print(json.dumps(dict(meta, loaded=herox.parameter_counts(params, critic)), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('convert', help='make a heroX checkpoint from a transformer-v3 policy and its critic')
    c.add_argument('--plato', default=str(ROOT / 'weights' / 'plato' / 'final.npz'))
    c.add_argument('--out', required=True)
    c.add_argument('--seed', type=int, default=0, help='seed of the new heads')
    c.set_defaults(run=convert)
    i = sub.add_parser('info', help='print a heroX checkpoint\'s metadata and parameter counts')
    i.add_argument('checkpoint')
    i.set_defaults(run=info)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
