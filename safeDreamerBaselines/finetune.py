"""Trains SafeDreamer's safe agent (BSRP-Lag) from a released reward-only agent, for a fixed number of simulator steps.

The question: if SafeDreamer's own safe training starts from the reward-only agent and not
from nothing, does the agent get under the cost limit within the budget, and what reward
does it keep?

Run it inside the image, with /datasets visible. From this folder:

  ../msCli/ms submit -p bigbatch -t 3d -n finetune-PointGoal1 -- apptainer exec --nv \\
      --bind /datasets/smohan --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \\
      python3 finetune.py --steps 250000 --seed 0 \\
      --out /datasets/smohan/safedreamer_finetune/PointGoal1_seed0 --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt

What it does:

  1. Builds the simulator and a new safe agent (variant bsrp_lag) from SafeDreamer's settings.
  2. Copies the reward-only checkpoint into the agent: every array the two agents share.
     What only the safe agent has (cost head, cost critic, Lagrange multipliers) keeps the
     values it was built with. See initialise().
  3. Hands the agent to SafeDreamer's own training loop, embodied.run.train_eval.
  4. Counts every simulator step where the simulator is called, and ends the loop when the
     budget is used up. See Meter.

A step is always one step of the simulator. A decision of the agent lasts 2 of them, as it
did when the reward-only agent was trained, so 250,000 steps are 125,000 decisions and 250
episodes. The first 2,200 or so steps are taken with random actions, which is how
SafeDreamer's loop fills its replay buffer before it trains. They are part of the budget.

It writes into the --out folder:

  run.json             the settings of the run, every SafeDreamer setting that differs from
                       SafeDreamer's own for bsrp_lag, and at the end the totals
  initialisation.json  which arrays were copied from the checkpoint and which were left fresh
  episodes.jsonl       one line per training episode: simulator steps so far, reward, cost
  checkpoints/         the agent after 0 steps, after every --checkpoint-every steps and at
                       the end, named like finetune-0250000_bsrp_lag_<task>_<seed>.ckpt
  safedreamer/         what SafeDreamer's loop writes itself: its settings, its training
                       numbers (metrics.jsonl) and one checkpoint of its own from the start

A checkpoint is evaluated with evaluate.py, which must be told that a decision lasts 2
simulator steps. From the file name alone it would take bsrp_lag's own 4:

  python3 evaluate.py --checkpoint .../finetune-0250000_bsrp_lag_..._0.ckpt --env.safetygym.repeat 2

Any option this script does not know is passed on to SafeDreamer's settings, as in
evaluate.py. A run cannot be continued. Started again in the same folder, it moves what an
unfinished run left into interrupted-N/ and starts from the reward-only checkpoint again.
A finished run is not repeated.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

# evaluate.py knows how a checkpoint is named, and importing it puts SafeDreamer's code on
# the path. recording.py has the way to make a folder on the cluster.
from evaluate import NAME_PATTERN
from recording import make_folder

VARIANT = 'bsrp_lag'

# evaluate.py reads the variant, the task and the seed from the end of this name.
CHECKPOINT_NAME = 'finetune-{steps:07d}_' + VARIANT + '_{task}_{seed}.ckpt'

# The two agents keep the actor, its optimiser and the scale of the returns under different
# names: SafeDreamer calls the actor-critic "ac" in the reward-only agent and "safe_ac" in
# the safe one. Without this the actor would not be copied.
RENAMED = {'agent/task_behavior/ac/': 'agent/task_behavior/safe_ac/'}

# The parts that only the safe agent has. Nothing else may be left fresh.
SAFE_ONLY = (
    'agent/wm/cost/',                          # the world model's cost head
    'agent/task_behavior/cost_critic/',        # the cost critic, its slow copy and its optimiser
    'agent/task_behavior/safe_ac/costnorm_',   # the scale of the cost returns
    'agent/task_behavior/safe_ac/lagrange/',   # the Lagrange multiplier and the penalty multiplier
)

# Episode k of a run is played in the world with seed FIRST_WORLD * (run seed + 1) + k. The
# evaluation plays worlds 0 to 49 and the adapter work stays below a million, so no world
# the agent trains in is one that anything is scored on.
FIRST_WORLD = 10000000

# Everything a run writes into its folder.
OUR_FILES = ('run.json', 'initialisation.json', 'episodes.jsonl', 'checkpoints', 'safedreamer')


def write_json(path, content):
    # Written under another name first, so that the file is never seen half-written.
    unfinished = path.with_name(path.name + '.part')
    unfinished.write_text(json.dumps(content, indent=1) + '\n')
    os.replace(unfinished, path)


def earlier_run(out):
    """Looks for an earlier run in the folder. True if it finished; an unfinished one is moved aside.

    The cluster starts a job again from the top when a node fails. A run cannot be continued
    from where it stopped (the replay buffer is not kept on disk), so what the cut-short
    attempt wrote is moved into interrupted-1/, interrupted-2/ and so on, and the run starts
    from the reward-only checkpoint again. Nothing is deleted.
    """
    left = [out / name for name in OUR_FILES if (out / name).exists()]
    if not left:
        return False
    run_file = out / 'run.json'
    if run_file.exists() and json.loads(run_file.read_text()).get('finished'):
        return True
    number = 1
    while (out / f'interrupted-{number}').exists():
        number += 1
    aside = out / f'interrupted-{number}'
    make_folder(aside)
    for path in left:
        path.rename(aside / path.name)
    print(f'An earlier run in {out} did not finish. Moved what it wrote to {aside}.', flush=True)
    return False


def safe_name(name):
    """The safe agent's name for what the reward-only agent calls `name`."""
    for theirs, ours in RENAMED.items():
        if name.startswith(theirs):
            return ours + name[len(theirs):]
    return name


def walk(value, visit, place=()):
    """Gives back `value` with every array in it replaced by visit(place, array).

    Most of an agent's variables are one array. An optimiser's state is several, nested in
    tuples and dictionaries. `place` says where an array sits: a tuple of names that starts
    with the variable's own name. Names from the reward-only agent are changed to the safe
    agent's on the way, so that the same array has the same place in both agents.
    """
    if isinstance(value, dict):
        return {name: walk(inner, visit, place + (safe_name(name),)) for name, inner in value.items()}
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        # A tuple with named entries, which is how the optimiser library keeps its states.
        return type(value)(*[walk(inner, visit, place + (name,)) for name, inner in zip(value._fields, value)])
    if isinstance(value, (tuple, list)):
        return type(value)(walk(inner, visit, place + (str(index),)) for index, inner in enumerate(value))
    if value is None:
        return None
    return visit(place, value)


def safe_only(place):
    return any(name.startswith(part) for name in place for part in SAFE_ONLY)


def initialise(agent, checkpoint_path, report_path):
    """Starts the safe agent from the reward-only checkpoint, and stops if the two do not fit as expected.

    Every array that both agents have, in the same place and with the same shape, is taken
    from the checkpoint: the world model with its encoder, decoder, reward head and continue
    head, the actor, the reward critic and its slow copy, the scale of the returns, and the
    optimiser state that belongs to those parts. Everything else keeps the value the safe
    agent was built with: the cost head, the cost critic and its slow copy, the scale of the
    cost returns, the Lagrange multiplier and the penalty multiplier, and the optimiser
    state that belongs to the cost head and the cost critic.

    Two things must hold, or nothing is trained. Every array in the checkpoint found its
    place: otherwise part of the reward-only agent would be dropped without a word. And
    every array left fresh belongs to a part that only the safe agent has: otherwise part
    of the agent would start from nothing when it should not.

    Writes the full list to `report_path` and returns the totals.
    """
    import embodied

    loaded = embodied.unpack(pathlib.Path(checkpoint_path).read_bytes())
    theirs = {}
    walk(loaded['agent'], lambda place, array: theirs.setdefault(place, array))

    copied, fresh = {}, {}

    def choose(place, ours):
        given = theirs.get(place)
        if given is not None and np.shape(given) == np.shape(ours) and np.asarray(given).dtype == np.asarray(ours).dtype:
            copied[place] = given
            return np.asarray(given)
        fresh[place] = ours
        return ours

    # agent.save() gives every variable of the newly built agent, and agent.load() takes
    # them back. They are SafeDreamer's own way in and out of a checkpoint.
    merged = walk(agent.save(), choose)
    no_place = {place: array for place, array in theirs.items() if place not in copied}
    unexpected = {place: array for place, array in fresh.items() if not safe_only(place)}

    def listed(arrays):
        # One line per array: where it sits, then its shape.
        return [f'{" > ".join(place)}  {tuple(np.shape(arrays[place]))}' for place in sorted(arrays)]

    def numbers(arrays):
        return int(sum(np.size(array) for array in arrays.values()))

    totals = {
        'arrays_copied': len(copied), 'numbers_copied': numbers(copied),
        'arrays_fresh': len(fresh), 'numbers_fresh': numbers(fresh),
    }
    write_json(report_path, {
        'checkpoint': pathlib.Path(checkpoint_path).name,
        # What SafeDreamer's own step counter read when the reward-only agent was saved.
        'checkpoint_step': int(loaded['step']) if 'step' in loaded else None,
        'renamed': RENAMED,
        **totals,
        'copied': listed(copied),
        'fresh': listed(fresh),
        'in_checkpoint_without_a_place': listed(no_place),
        'fresh_outside_the_safe_only_parts': listed(unexpected),
    })

    # One line per variable. An optimiser's state counts as one variable of many arrays.
    variables = {}
    for place in copied:
        variables.setdefault(place[0], [0, 0])[0] += 1
    for place in fresh:
        variables.setdefault(place[0], [0, 0])[1] += 1
    for name, (from_checkpoint, left_fresh) in sorted(variables.items()):
        if not left_fresh:
            print(f'  copied  {name}')
        elif not from_checkpoint:
            print(f'  fresh   {name}')
        else:
            print(f'  partly  {name}: {from_checkpoint} arrays copied, {left_fresh} fresh')
    print(f'Copied {len(copied)} arrays ({totals["numbers_copied"]:,} numbers) from the checkpoint, '
          f'left {len(fresh)} fresh ({totals["numbers_fresh"]:,} numbers). Full list in {report_path}', flush=True)

    if no_place or unexpected:
        raise SystemExit(
            f'The checkpoint does not fit the safe agent as expected. {len(copied)} arrays were copied, but '
            f'{len(no_place)} arrays of the checkpoint have no place in the safe agent, e.g. {listed(no_place)[:3]}, '
            f'and {len(unexpected)} arrays outside the safe-only parts would be left fresh, e.g. '
            f'{listed(unexpected)[:3]}. Look at RENAMED and SAFE_ONLY at the top of finetune.py.')

    agent.load(merged)
    return totals


class BudgetReached(Exception):
    """Raised where the simulator is called, once the budget of simulator steps is used up."""


class Meter:
    """Counts the simulator steps of a training run where they happen: at the simulator itself.

    SafeDreamer's own counter counts something else, one for every observation the agent is
    handed. That is one per decision, which lasts several simulator steps, and one more for
    the observation at every reset, which costs none. So the budget is kept here. The
    simulator's reset and step are replaced by versions that

        give every episode its own seed, so the worlds of a run can be played again;
        add up the reward and the cost of the running episode, and write a line when it ends;
        save a checkpoint when the count reaches one of the chosen numbers;
        refuse to go on once the budget is used up, which is what ends SafeDreamer's loop.

    A checkpoint is saved at the next call after its step, not during the step. By then the
    loop has finished with that step, any training it sets off included, so every
    checkpoint is taken at the same point of the loop.
    """

    def __init__(self, simulator, budget, save_at, first_world, episodes_path, save, tell):
        self.budget = budget
        self.save_at = set(save_at)
        self.episodes_path = episodes_path
        self.save = save
        self.tell = tell
        self.simulator_steps = 0
        self.random_steps = 0
        self.episodes = 0
        self.cost = 0.0
        self.agent_playing = False   # main() sets this the first time the agent chooses an action
        self.started = time.perf_counter()
        self._episode = None

        unhooked_reset, unhooked_step = simulator.reset, simulator.step

        def reset(**_):
            self._between_steps()
            self.close()   # SafeDreamer's loop only resets after an episode's end, so this writes nothing
            self._episode = {'episode': self.episodes, 'seed': first_world + self.episodes,
                             'steps': 0, 'random_steps': 0, 'reward': 0.0, 'cost': 0.0}
            self.episodes += 1
            return unhooked_reset(seed=self._episode['seed'])

        def step(action):
            self._between_steps()
            result = unhooked_step(action)
            _, reward, cost, terminated, truncated, _ = result
            self.simulator_steps += 1
            self._episode['steps'] += 1
            self._episode['reward'] += float(reward)
            self._episode['cost'] += float(cost)
            if not self.agent_playing:
                self._episode['random_steps'] += 1
            if terminated or truncated:
                self._end_episode(finished=True)
            return result

        simulator.reset, simulator.step = reset, step

    def _between_steps(self):
        if self.simulator_steps in self.save_at:
            self.save_at.remove(self.simulator_steps)
            self.save(self.simulator_steps)
        if self.simulator_steps >= self.budget:
            raise BudgetReached

    def _end_episode(self, finished):
        episode, self._episode = self._episode, None
        self.random_steps += episode['random_steps']
        self.cost += episode['cost']
        line = {
            'episode': episode['episode'],
            'seed': episode['seed'],
            # Counted from the start of the run, this episode included.
            'simulator_steps': self.simulator_steps,
            'episode_steps': episode['steps'],
            'reward': episode['reward'],
            'cost': episode['cost'],
            # How many of this episode's steps were taken with random actions, while
            # SafeDreamer's loop filled its replay buffer. Zero once the agent plays.
            'random_steps': episode['random_steps'],
            # False only for an episode that was cut short, as the last one is if the
            # budget ends in the middle of it.
            'finished': finished,
            'seconds': round(time.perf_counter() - self.started, 1),
        }
        with self.episodes_path.open('a') as file:
            file.write(json.dumps(line) + '\n')
        print(f'episode {line["episode"]}: {self.simulator_steps:,} of {self.budget:,} simulator steps, '
              f'reward {line["reward"]:.2f}, cost {line["cost"]:.0f}', flush=True)
        if self.tell:
            self.tell(step=self.simulator_steps, of=self.budget, episode=line['episode'], cost=line['cost'])

    def close(self):
        """Writes the line of an episode that is still running, as one is if the budget ends in the middle of it."""
        if self._episode and self._episode['steps']:
            self._end_episode(finished=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True, help='the reward-only (dreamerv3) checkpoint to start from')
    parser.add_argument('--steps', type=int, default=250000,
                        help='simulator steps to train for, the random ones that fill the replay buffer included')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', help='folder for everything the run writes (default: finetune/ in the run folder)')
    parser.add_argument('--checkpoint-every', type=int, default=50000,
                        help='save the agent each time this many more simulator steps have been taken')
    args, safedreamer_options = parser.parse_known_args()
    if args.steps < 1 or args.checkpoint_every < 1 or args.seed < 0:
        parser.error('--steps and --checkpoint-every must be at least 1, and --seed at least 0')
    if FIRST_WORLD * (args.seed + 1) + args.steps >= 2 ** 32:
        parser.error('--seed is too large: the simulator takes world seeds below 2 to the power 32')

    match = NAME_PATTERN.search(args.checkpoint)
    if not match or match.group(1) != 'dreamerv3':
        raise SystemExit(f'{args.checkpoint} is not named like a reward-only (dreamerv3) checkpoint')
    if not pathlib.Path(args.checkpoint).is_file():
        raise SystemExit(f'There is no file {args.checkpoint}')
    task = match.group(2)

    out = pathlib.Path(args.out or pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'finetune')
    make_folder(out)
    if earlier_run(out):
        print(f'{out} already holds a finished run')
        return
    checkpoints, logdir = out / 'checkpoints', out / 'safedreamer'
    make_folder(checkpoints)
    make_folder(logdir)

    tell = None
    if 'MS_RUN_DIR' in os.environ:
        # In a job, `ms status` shows the last thing written to the run folder's status file.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'msCli'))
        from ms_status import write_status
        tell = write_status

    # Imported here, not at the top, because importing SafeDreamer starts up JAX and TensorFlow.
    import SafeDreamer  # noqa: F401  (also makes `embodied` importable)
    import embodied
    from embodied.envs.safetygym import SafetyGym
    from SafeDreamer import agent as agt
    from SafeDreamer import train as safedreamer_train

    # The settings, built the way SafeDreamer's own entry script builds them for bsrp_lag,
    # with the few changes below. Every one is explained in finetune_notes.md.
    untouched = embodied.Config(agt.Agent.configs['defaults']).update(agt.Agent.configs[VARIANT])
    reward_only = embodied.Config(agt.Agent.configs['defaults']).update(agt.Agent.configs['dreamerv3'])
    config = embodied.Flags(untouched).parse([
        '--method', VARIANT, '--task', task, '--seed', str(args.seed), '--logdir', str(logdir),
        # A decision lasts as many simulator steps as it did when the reward-only agent was
        # trained. bsrp_lag's own setting is twice that, which the world model never saw.
        '--env.safetygym.repeat', str(reward_only.env.safetygym.repeat),
        # No evaluation episodes during training: they would be simulator steps outside
        # the budget. The checkpoints are evaluated afterwards.
        '--run.eval_every', '0',
        # SafeDreamer's own checkpoint, written every 15 minutes, is not needed.
        '--run.save_every', '0',
    ] + safedreamer_options)
    steps_per_decision = config.env.safetygym.repeat
    theirs = untouched.flat
    changed = {name: [theirs[name], value] for name, value in config.flat.items() if theirs[name] != value}
    # SafeDreamer's loop has a limit of its own, which counts observations: at most two for
    # every simulator step. It must not end the run before the budget does.
    if config.run.steps <= 2 * args.steps + 1:
        raise SystemExit(f'--run.steps {config.run.steps} would end the run before {args.steps} simulator steps')

    run = {
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'variant': VARIANT,
        'task': task,
        'seed': args.seed,
        'budget_simulator_steps': args.steps,
        'checkpoint_every': args.checkpoint_every,
        'simulator_steps_per_decision': steps_per_decision,
        'first_world_seed': FIRST_WORLD * (args.seed + 1),
        'rendering': os.environ.get('MUJOCO_GL'),
        # For each setting: SafeDreamer's own value for bsrp_lag, then the value used.
        'settings_changed': changed,
        'finished': False,
    }
    write_json(out / 'run.json', run)
    print('Settings that differ from SafeDreamer\'s own for bsrp_lag:')
    for name, (before, after) in sorted(changed.items()):
        print(f'  {name}: {before} -> {after}')

    # The simulator, built as in evaluate.py. It runs inside this process, not in a process
    # of its own as SafeDreamer's training script has it, so that the meter can reach it.
    simulator_wrapper = SafetyGym(
        task.split('_', 1)[1], platform=config.jax.platform, **config.env.safetygym)
    simulator = simulator_wrapper._dmenv
    env = embodied.BatchEnv([safedreamer_train.wrap_env(simulator_wrapper, config)], parallel=False)

    # SafeDreamer's step counter starts at zero: it counts this run, not the reward-only
    # agent's training before it.
    safedreamer_step = embodied.Counter()
    agent = agt.Agent(env.obs_space, env.act_space, safedreamer_step, config)
    print(f'Initialising the {VARIANT} agent from {args.checkpoint}', flush=True)
    run['initialisation'] = initialise(agent, args.checkpoint, out / 'initialisation.json')

    # The rest of what SafeDreamer's training script sets up. The logger is its own without
    # the TensorBoard files, which would hold a video of every logged episode. The replay
    # buffer is its own, kept in memory and not also written to disk.
    config.save(logdir / 'config.yaml')
    logger = embodied.Logger(safedreamer_step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score|episode/cost'),
    ], steps_per_decision)
    replay = safedreamer_train.make_replay(config)
    run_settings = embodied.Config(
        **config.run, logdir=str(logdir), use_cost=config.use_cost,
        batch_steps=config.batch_size * config.batch_length)
    # The loop asks for this second, PID-controlled multiplier and keeps it up to date, but
    # only the variant that plans with it (osrp_lag) reads it. BSRP-Lag's own multipliers
    # are variables of the agent.
    pid_lagrangian = safedreamer_train.PIDLagrangian(config)

    def save(simulator_steps):
        name = CHECKPOINT_NAME.format(steps=simulator_steps, task=task, seed=args.seed)
        # Written under another name first, so that a file with the final name is always complete.
        unfinished = checkpoints / (name + '.part')
        saver = embodied.Checkpoint(unfinished, log=False, parallel=False)
        saver.agent = agent
        # evaluate.py reads this number as the decisions made in training. SafeDreamer's own
        # counter would be a little more: it also counts the observation at every reset.
        saver.step = embodied.Counter(simulator_steps // steps_per_decision)
        saver.save()
        unfinished.rename(checkpoints / name)
        print(f'Saved {checkpoints / name}', flush=True)

    save_at = set(range(0, args.steps, args.checkpoint_every)) | {args.steps}
    meter = Meter(simulator, args.steps, save_at, run['first_world_seed'], out / 'episodes.jsonl', save, tell)

    # The loop first plays with random actions to fill its replay buffer. The meter is told
    # when the agent takes over, so that it can count the random steps apart.
    play = agent.policy

    def policy(*inputs, **options):
        meter.agent_playing = True
        return play(*inputs, **options)

    agent.policy = policy

    # SafeDreamer's own loop. It wants a second simulator and replay buffer for evaluation
    # episodes. With evaluation switched off it never steps or fills them, so the training
    # ones are handed over in their place. The loop has no exit at an exact step, so the
    # meter ends it: it refuses the first simulator call after the budget.
    try:
        embodied.run.train_eval(agent, env, env, replay, replay, logger, run_settings, pid_lagrangian)
    except BudgetReached:
        pass
    else:
        raise SystemExit('SafeDreamer\'s loop ended before the budget was used up')
    meter.close()

    updates = getattr(agent, '_updates', None)
    run.update({
        'finished': True,
        'simulator_steps': meter.simulator_steps,
        'random_simulator_steps': meter.random_steps,
        'episodes': meter.episodes,
        # The cost paid while training: the sum over every line of episodes.jsonl.
        'cost_in_training': meter.cost,
        'training_updates': None if updates is None else int(updates),
        'safedreamer_step_counter': int(safedreamer_step),
        'seconds': round(time.perf_counter() - meter.started, 1),
    })
    write_json(out / 'run.json', run)
    print(f'Finished: {meter.simulator_steps:,} simulator steps ({meter.random_steps:,} of them with random actions), '
          f'{meter.episodes} episodes, {run["training_updates"]} training updates, '
          f'cost in training {meter.cost:.0f}. Everything is in {out}', flush=True)
    if not meter.agent_playing:
        print('The budget ended while the replay buffer was still being filled with random actions, '
              'so the agent was never trained.', flush=True)
    # What SafeDreamer's script does when its loop ends.
    logger.write()
    env.close()


if __name__ == '__main__':
    main()
