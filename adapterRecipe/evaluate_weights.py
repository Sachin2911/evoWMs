"""Plays an adapter whose weights are set by hand, to check what its inputs can express.

  python3 evaluate_weights.py --checkpoint X.ckpt --cost-net cost_net.npz --reads look --gate ahead15 \\
      --set 4,3=3 2,3=-3 --out FOLDER

This is a check on the adapter's inputs, not a way of making adapters: it answers "is there
any setting of the weights that steers the agent around hazards?" before evolution is asked
to find one. Each --set entry is INPUT,OUTPUT=VALUE. OUTPUT counts the gains and then the
shifts (for a 2-number action: 0 and 1 are the gains, 2 and 3 the shifts). INPUT is the
number of the input, or `bias` for the constant. VALUE is what that input contributes per
unit, so it does not depend on how many inputs there are.

Writes adapter.npz and episodes.jsonl in FOLDER, playing through evaluate_adapter.py.
"""

import argparse
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cost-net', required=True)
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--reads', choices=adapter.READS, required=True)
    parser.add_argument('--gate', choices=adapter.GATES, required=True)
    parser.add_argument('--set', nargs='*', default=[])
    parser.add_argument('--action-size', type=int, default=2)
    parser.add_argument('--output-bound', type=float, default=4.0)
    parser.add_argument('--look-horizon', type=int, default=8)
    parser.add_argument('--look-samples', type=int, default=4)
    parser.add_argument('--look-push', type=float, default=1.0)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--first-seed', type=int, default=3000)
    args = parser.parse_args()

    net = dict(np.load(args.cost_net))
    count = adapter.count_features(net, args.reads, args.action_size)
    outputs = 2 * args.action_size
    weights = np.zeros((count + 1) * outputs, np.float32)
    for entry in args.set:
        place, value = entry.split('=')
        feature, output = place.split(',')
        if feature == 'bias':
            weights[count * outputs + int(output)] = float(value)
        else:
            # adapter.adapt divides the inputs' weights by the square root of their number.
            weights[int(feature) * outputs + int(output)] = float(value) * np.sqrt(count)
    args.out.mkdir(parents=True, exist_ok=True)
    adapter.save(args.out / 'adapter.npz', weights, args.reads, args.gate, net, args.output_bound,
                 look_horizon=args.look_horizon, look_samples=args.look_samples, look_push=args.look_push)
    os.execv(sys.executable, [sys.executable, str(HERE / 'evaluate_adapter.py'), '--checkpoint', args.checkpoint,
                              '--adapter', str(args.out / 'adapter.npz'), '--out', str(args.out / 'episodes.jsonl'),
                              '--episodes', str(args.episodes), '--first-seed', str(args.first_seed)])


if __name__ == '__main__':
    main()
