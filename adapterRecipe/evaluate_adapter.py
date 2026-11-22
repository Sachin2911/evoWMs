"""Plays a kept adapter on its frozen agent for a number of fixed-seed episodes.

  python3 evaluate_adapter.py --checkpoint X.ckpt --adapter adapter_penalty.npz --out episodes.jsonl

Episode k uses seed first-seed + k for both the world layout and the agent's random
choices, as safeDreamerBaselines/evaluate.py does for the baselines. One line per episode.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safeDreamerBaselines'))
sys.path.insert(0, str(HERE))

import adapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--first-seed', type=int, default=0)
    args, safedreamer_options = parser.parse_known_args()

    import evaluate
    built = evaluate.build(args.checkpoint, safedreamer_options)
    import embodied
    weights, reads, gate, net, bound = adapter.load(args.adapter)
    look, on_start, look_rng = None, None, {}
    if reads in adapter.LOOKS:
        # The look-ahead runs inside the world model; evolve.py builds it the same way.
        import evolve
        import jax
        notes = np.load(args.adapter)
        push = float(notes['look_push']) if 'look_push' in notes.files else 1.0
        looking = evolve.make_look(built, int(notes['look_horizon']), int(notes['look_samples']), push)
        # An adapter kept by a run with --look-with-cost-model carries the network its look-ahead asks.
        look_net = {name[len('looknet_'):]: notes[name] for name in notes.files if name.startswith('looknet_')} or net
        on_gpu = jax.device_put({name: np.asarray(value, np.float32) for name, value in look_net.items()}, built.agent.policy_devices[0])

        def look(latent, action):
            return looking(latent, action, on_gpu, int(look_rng['now'].integers(2 ** 31)))

        def on_start(seed):
            look_rng['now'] = np.random.default_rng(seed + 7919)

    done = len(evaluate.read_results(args.out))
    seeds = iter(range(args.first_seed + done, args.first_seed + args.episodes))
    current = evaluate.seed_episodes(built.simulator, built.agent, seeds, on_start)

    def record(episode, _worker):
        result = {'seed': current['seed'], 'reward': float(episode['reward'].astype(np.float64).sum()),
                  'cost': float(episode['cost'].astype(np.float64).sum()), 'adapter': pathlib.Path(args.adapter).name}
        with args.out.open('a') as file:
            file.write(json.dumps(result) + '\n')
        print(result, flush=True)

    driver = embodied.Driver(built.env)
    driver.on_episode(record)
    if args.episodes > done:
        driver(adapter.adapted_policy(built.agent, net, reads, gate, lambda: weights, bound=bound, look=look), episodes=args.episodes - done, lag=built.penalty)
    results = evaluate.read_results(args.out)
    reward = np.array([result['reward'] for result in results])
    cost = np.array([result['cost'] for result in results])
    print(f'{len(results)} episodes: reward {reward.mean():.2f} (sd {reward.std():.2f}), cost {cost.mean():.2f} '
          f'(sd {cost.std():.2f}), {int((cost <= 2).sum())} within the limit')


if __name__ == '__main__':
    main()
