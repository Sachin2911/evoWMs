"""Plays one released SafeDreamer checkpoint for a number of fixed-seed episodes.

Writes one line per episode (seed, reward, cost, length) to a results file.
Run it inside the image, with /datasets visible:

  apptainer exec --nv --bind /datasets/$USER ~/images/safeDreamer.sif \\
      python3 evaluate.py --checkpoint /datasets/$USER/safedreamer_checkpoints/.../X.ckpt

The variant and task are read from the checkpoint's file name. Any option this script
does not know is passed on to SafeDreamer's own settings, e.g. --env.safetygym.repeat 4.

With --record FOLDER it also saves every simulator step and the layout of each world, and
with --video-seeds a video of the chosen episodes; see recording.py. Everything for the
checkpoint, the results file included, then goes in FOLDER/<checkpoint name>/. With
--record-latents it also saves the world model's internal state at every decision.

With --adapter FILE the agent's actions are changed by an evolved adapter (see adapterPilot).
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time
import types

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

    Returns how many decisions the agent had made in training when the checkpoint was saved.
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
    training_decisions = embodied.Counter()
    checkpoint.step = training_decisions
    checkpoint.load(checkpoint_path, keys=['agent', 'step'])
    loaded = shapes()
    if loaded != expected:
        different = sorted(name for name in expected.keys() | loaded.keys()
                           if expected.get(name) != loaded.get(name))
        raise SystemExit(
            f'Checkpoint does not match the settings: {len(different)} of {len(expected)} '
            f'variables differ, e.g. {different[:5]}')
    return int(training_decisions)


def read_name(checkpoint_path):
    """The variant, task and training seed in a checkpoint's file name."""
    match = NAME_PATTERN.search(str(checkpoint_path))
    if not match:
        raise SystemExit(f'Cannot read the variant and task from the name of {checkpoint_path}')
    return match.groups()


def build(checkpoint_path, safedreamer_options=()):
    """Builds the simulator and the agent for a checkpoint, and loads the checkpoint into the agent."""
    variant, task, training_seed = read_name(checkpoint_path)

    # Imported here, not at the top, because importing SafeDreamer starts up JAX and TensorFlow.
    import SafeDreamer  # noqa: F401  (also makes `embodied` importable)
    import embodied
    from embodied.envs.safetygym import SafetyGym
    from SafeDreamer import agent as agt
    from SafeDreamer import train as safedreamer_train

    # The settings, built the way SafeDreamer's own entry script builds them.
    config = embodied.Config(agt.Agent.configs['defaults']).update(agt.Agent.configs[variant])
    config = embodied.Flags(config).parse(
        ['--method', variant, '--task', task] + VARIANT_SETTINGS.get(variant, []) + list(safedreamer_options))

    # The simulator, with the camera set-up the agent was trained on (two 64x64 images).
    simulator_wrapper = SafetyGym(
        task.split('_', 1)[1], platform=config.jax.platform, **config.env.safetygym)
    env = embodied.BatchEnv([safedreamer_train.wrap_env(simulator_wrapper, config)], parallel=False)

    agent = agt.Agent(env.obs_space, env.act_space, embodied.Counter(), config)
    training_decisions = load_checkpoint(agent, checkpoint_path)
    return types.SimpleNamespace(
        agent=agent, env=env, simulator=simulator_wrapper._dmenv, config=config, variant=variant, task=task,
        training_seed=int(training_seed), training_decisions=training_decisions,
        steps_per_decision=config.env.safetygym.repeat,
        penalty=safedreamer_train.PIDLagrangian(config).lagrange_penalty)


def seed_episodes(simulator, agent, seeds, on_start=None):
    """Makes every new episode start from the next seed in `seeds`.

    The seed fixes both the layout of the world and the agent's random choices. Episode k is
    then the same whichever checkpoint plays it, and however many episodes came before it.
    SafeDreamer starts each episode by calling the simulator's reset, so that is where the
    seed is set. Returns a dict that always holds the current episode's seed and start time.
    """
    current = {}
    unseeded_reset = simulator.reset

    def seeded_reset(**_):
        current['seed'] = next(seeds)
        current['start'] = time.perf_counter()
        agent.rng = np.random.default_rng(current['seed'])
        result = unseeded_reset(seed=current['seed'])
        if on_start:
            on_start(current['seed'])
        return result

    simulator.reset = seeded_reset
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--first-seed', type=int, default=0,
                        help='episode k uses seed first-seed + k')
    parser.add_argument('--out', help='results file (default: episodes.jsonl in the run folder)')
    parser.add_argument('--record', help='folder to record every step in; each checkpoint gets a sub-folder')
    parser.add_argument('--video-seeds', type=int, nargs='*', default=[],
                        help='with --record: also save a video of the episodes with these seeds')
    parser.add_argument('--record-latents', action='store_true',
                        help="with --record: also save the world model's internal state at every decision")
    parser.add_argument('--adapter', help="change the agent's actions with this evolved adapter (adapterPilot)")
    args, safedreamer_options = parser.parse_known_args()
    if (args.video_seeds or args.record_latents) and not args.record:
        parser.error('--video-seeds and --record-latents need --record')

    variant, task, training_seed = read_name(args.checkpoint)

    record_folder = None
    if args.record:
        from recording import Recorder, make_folder
        record_folder = pathlib.Path(args.record) / pathlib.Path(args.checkpoint).stem
        make_folder(record_folder)
    if args.out:
        out_path = pathlib.Path(args.out)
    elif record_folder:
        out_path = record_folder / 'episodes.jsonl'
    else:
        out_path = pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'episodes.jsonl'
    # Episodes already in the results file are not played again, so a restarted job carries on.
    done = len(read_results(out_path))
    if done >= args.episodes:
        print(f'{out_path} already has {done} episodes')
        return

    built = build(args.checkpoint, safedreamer_options)
    agent, env, simulator, config = built.agent, built.env, built.simulator, built.config
    training_decisions = built.training_decisions

    recorder = None
    if record_folder:
        steps_per_decision = config.env.safetygym.repeat
        recorder = Recorder(simulator, record_folder, steps_per_decision, args.video_seeds)
        (record_folder / 'run.json').write_text(json.dumps({
            'checkpoint': pathlib.Path(args.checkpoint).name,
            'variant': variant,
            'task': task,
            'training_seed': int(training_seed),
            'simulator_steps_per_decision': steps_per_decision,
            # The checkpoint counts decisions; each one lasts several simulator steps.
            'training_decisions': training_decisions,
            'training_simulator_steps': training_decisions * steps_per_decision,
        }) + '\n')

    # Every episode gets its own seed: episode k uses first-seed + k.
    seeds = iter(range(args.first_seed + done, args.first_seed + args.episodes))
    current = seed_episodes(simulator, agent, seeds, recorder.start_episode if recorder else None)

    if recorder:
        # The recorder looks at the simulator after each of its steps, hooked in the same way.
        unrecorded_step = simulator.step

        def recorded_step(action):
            result = unrecorded_step(action)
            recorder.after_step(action, result)
            return result

        simulator.step = recorded_step

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
        if args.adapter:
            result['adapter'] = pathlib.Path(args.adapter).name
        if recorder:
            # Before the results line, so an episode counted as done always has its recording.
            recorder.finish_episode(episode)
        with out_path.open('a') as file:
            file.write(json.dumps(result) + '\n')
        print(result, flush=True)

    import embodied
    driver = embodied.Driver(env)
    driver.on_episode(record)

    if args.adapter:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'adapterPilot'))
        import adapter
        weights, kind, net = adapter.load(args.adapter)
        play = adapter.adapted_policy(agent, net, kind, lambda: weights)
    else:
        def play(*inputs):
            return agent.policy(*inputs, mode='eval')

    def policy(*inputs):
        actions, state = play(*inputs)
        if args.record_latents:
            recorder.after_decision(state)
        return actions, state

    driver(policy, episodes=args.episodes - done, lag=built.penalty)

    results = read_results(out_path)
    rewards = np.array([r['reward'] for r in results])
    costs = np.array([r['cost'] for r in results])
    print(f'{len(results)} episodes: reward {rewards.mean():.2f} (sd {rewards.std():.2f}), '
          f'cost {costs.mean():.2f} (sd {costs.std():.2f})')


if __name__ == '__main__':
    main()
