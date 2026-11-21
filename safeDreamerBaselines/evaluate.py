"""Plays one released SafeDreamer checkpoint for a number of fixed-seed episodes.

Writes one line per episode (seed, reward, cost, length) to a results file.
Run it inside the image, with /datasets visible:

  apptainer exec --nv --bind /datasets/$USER ~/images/safeDreamer.sif \\
      python3 evaluate.py --checkpoint /datasets/$USER/safedreamer_checkpoints/.../X.ckpt

The variant and task are read from the checkpoint's file name. Any option this script
does not know is passed on to SafeDreamer's own settings, e.g. --env.safetygym.repeat 4.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

# Where SafeDreamer's code is. The image has it at /opt/SafeDreamer.
sys.path.insert(0, os.environ.get('SAFEDREAMER_DIR', '/opt/SafeDreamer'))

# A checkpoint is named like 20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt:
# when it was saved, the variant, the task, and the seed it was trained with.
NAME_PATTERN = re.compile(r'_(dreamerv3|bsrp_lag|osrp_lag|osrp)_(safetygym_\w+-v0)_(\d+)\.ckpt$')

# Settings a variant needs at evaluation time that are not in its checkpoint.
# OSRP-Lag's Lagrange multiplier is one; 0.1 is the value in SafeDreamer's own evaluation command.
VARIANT_SETTINGS = {'osrp_lag': ['--pid.init_penalty', '0.1']}


def read_results(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_checkpoint(agent, checkpoint_path):
    """Loads the checkpoint, and stops if it does not fit the agent that the settings built.

    SafeDreamer's own loading does no such check: it would accept a checkpoint from a
    different variant or network size without complaint.
    """
    import embodied
    import jax

    def shapes():
        # Most variables are one array, but an optimiser's state is a bundle of several,
        # so each variable is described by the shapes of all the arrays inside it.
        return {name: [np.shape(array) for array in jax.tree_util.tree_leaves(value)]
                for name, value in agent.varibs.items()}

    expected = shapes()
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(checkpoint_path, keys=['agent'])
    loaded = shapes()
    if loaded != expected:
        different = sorted(name for name in expected.keys() | loaded.keys()
                           if expected.get(name) != loaded.get(name))
        raise SystemExit(
            f'Checkpoint does not match the settings: {len(different)} of {len(expected)} '
            f'variables differ, e.g. {different[:5]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--first-seed', type=int, default=0,
                        help='episode k uses seed first-seed + k')
    parser.add_argument('--out', help='results file (default: episodes.jsonl in the run folder)')
    args, safedreamer_options = parser.parse_known_args()

    match = NAME_PATTERN.search(args.checkpoint)
    if not match:
        raise SystemExit(f'Cannot read the variant and task from the name of {args.checkpoint}')
    variant, task, _ = match.groups()

    out_path = pathlib.Path(args.out or os.path.join(os.environ.get('MS_RUN_DIR', '.'), 'episodes.jsonl'))
    # Episodes already in the results file are not played again, so a restarted job carries on.
    done = len(read_results(out_path))
    if done >= args.episodes:
        print(f'{out_path} already has {done} episodes')
        return

    # Imported here, not at the top, because importing SafeDreamer starts up JAX and TensorFlow.
    import SafeDreamer  # noqa: F401  (also makes `embodied` importable)
    import embodied
    from embodied.envs.safetygym import SafetyGym
    from SafeDreamer import agent as agt
    from SafeDreamer import train as safedreamer_train

    # The settings, built the way SafeDreamer's own entry script builds them.
    config = embodied.Config(agt.Agent.configs['defaults']).update(agt.Agent.configs[variant])
    config = embodied.Flags(config).parse(
        ['--method', variant, '--task', task] + VARIANT_SETTINGS.get(variant, []) + safedreamer_options)

    # The simulator, with the camera set-up the agent was trained on (two 64x64 images).
    simulator_wrapper = SafetyGym(
        task.split('_', 1)[1], platform=config.jax.platform, **config.env.safetygym)
    simulator = simulator_wrapper._dmenv
    env = embodied.BatchEnv([safedreamer_train.wrap_env(simulator_wrapper, config)], parallel=False)

    agent = agt.Agent(env.obs_space, env.act_space, embodied.Counter(), config)
    load_checkpoint(agent, args.checkpoint)

    # Every episode gets its own seed, which fixes both the layout of the world and the
    # agent's random choices. Episode k is then the same whichever checkpoint plays it, and
    # however many episodes came before it. SafeDreamer starts each episode by calling the
    # simulator's reset, so that is where the seed is set.
    seeds = iter(range(args.first_seed + done, args.first_seed + args.episodes))
    current = {}
    unseeded_reset = simulator.reset

    def seeded_reset(**_):
        current['seed'] = next(seeds)
        current['start'] = time.perf_counter()
        agent.rng = np.random.default_rng(current['seed'])
        return unseeded_reset(seed=current['seed'])

    simulator.reset = seeded_reset

    def record(episode, _worker):
        result = {
            'checkpoint': pathlib.Path(args.checkpoint).name,
            'variant': variant,
            'task': task,
            'seed': current['seed'],
            'reward': float(episode['reward'].astype(np.float64).sum()),
            'cost': float(episode['cost'].astype(np.float64).sum()),
            # The first entry is the observation at reset, before any decision.
            'decisions': len(episode['reward']) - 1,
            'seconds': round(time.perf_counter() - current['start'], 1),
        }
        with out_path.open('a') as file:
            file.write(json.dumps(result) + '\n')
        print(result, flush=True)

    driver = embodied.Driver(env)
    driver.on_episode(record)
    penalty = safedreamer_train.PIDLagrangian(config).lagrange_penalty
    driver(lambda *inputs: agent.policy(*inputs, mode='eval'),
           episodes=args.episodes - done, lag=penalty)

    results = read_results(out_path)
    rewards = np.array([r['reward'] for r in results])
    costs = np.array([r['cost'] for r in results])
    print(f'{len(results)} episodes: reward {rewards.mean():.2f} (sd {rewards.std():.2f}), '
          f'cost {costs.mean():.2f} (sd {costs.std():.2f})')


if __name__ == '__main__':
    main()
