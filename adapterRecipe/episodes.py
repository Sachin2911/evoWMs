"""Plays seeded episodes of a frozen reward-only agent, alone or with an adapter, and saves them.

An episode is kept as arrays with one entry per decision. Entry t belongs to the decision
made on seeing observation t:

    deter, stoch   the world model's internal state after observation t: its 512 continuous
                   numbers, and its 48 categorical choices as the number of each choice
    proposed       the action the frozen agent proposed
    action         the action taken, which is the same unless an adapter changed it
    reward, cost   what came with observation t. They are the result of the action of
                   entry t - 1, added up over the simulator steps that one decision lasts

The first entry is the observation at reset, where reward and cost are 0. The last entry is
the observation that ends the episode: its action is never taken, and is saved as zeros.

With Player.watch set, an episode also holds what the simulator itself reported, again as
what came of the action of entry t - 1:

    cost_<kind>    the cost charged by each kind of object (hazards, buttons, ...)
    speed          how fast the robot moved, as the mean over the decision's simulator steps
    x, y           where the robot was at the end of the decision
    goals          how many times the goal was reached during the decision
    near_<kind>    the distance from the robot to the nearest object of each kind

Run as a script, it plays the agent with no adapter and saves one file per episode. From
this folder:

  ../msCli/ms submit -t 1h -n episodes -- apptainer exec --nv --bind /datasets/smohan \\
      /home-mscluster/smohan/images/safeDreamer.sif python3 episodes.py --episodes 25 --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

# evaluate.py has the check that a checkpoint fits the agent, and recording.py the way to
# make a folder on the cluster. Importing evaluate also puts SafeDreamer's code on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'safeDreamerBaselines'))
from evaluate import NAME_PATTERN, load_checkpoint  # noqa: E402
from recording import make_folder  # noqa: E402
from frozen_agent import FrozenDreamer  # noqa: E402


def states(episode, entries=slice(None)):
    """The internal states at the chosen entries of an episode, in the form the pieces of frozen_agent.py take."""
    # The pieces want each categorical choice written out as zeros and a one.
    written_out = np.eye(int(episode['classes']), dtype=np.float16)[episode['stoch'][entries]]
    return {'deter': episode['deter'][entries], 'stoch': written_out}


class Player:
    """A frozen reward-only agent with its simulator, ready to play seeded episodes."""

    def __init__(self, checkpoint):
        match = NAME_PATTERN.search(str(checkpoint))
        if not match or match.group(1) != 'dreamerv3':
            raise SystemExit(f'{checkpoint} is not named like a reward-only (dreamerv3) checkpoint')
        variant, task, _ = match.groups()

        # The agent and its simulator, built the way evaluate.py builds them.
        import SafeDreamer  # noqa: F401  (also makes `embodied` importable)
        import embodied
        from embodied.envs.safetygym import SafetyGym
        from SafeDreamer import agent as agt
        from SafeDreamer import train as safedreamer_train

        config = embodied.Config(agt.Agent.configs['defaults']).update(agt.Agent.configs[variant])
        config = embodied.Flags(config).parse(['--method', variant, '--task', task])
        simulator_wrapper = SafetyGym(
            task.split('_', 1)[1], platform=config.jax.platform, **config.env.safetygym)
        simulator = simulator_wrapper._dmenv
        self._simulator_wrapper = simulator_wrapper
        self._env = embodied.BatchEnv(
            [safedreamer_train.wrap_env(simulator_wrapper, config)], parallel=False)
        self.act_space = self._env.act_space
        self.agent = agt.Agent(self._env.obs_space, self._env.act_space, embodied.Counter(), config)
        load_checkpoint(self.agent, checkpoint)
        self.frozen = FrozenDreamer(self.agent)
        self.task = task
        self.steps_per_decision = config.env.safetygym.repeat
        self._driver = embodied.Driver
        self._penalty = safedreamer_train.PIDLagrangian(config).lagrange_penalty

        # As in evaluate.py, the seed of an episode is set where SafeDreamer starts it: in
        # the simulator's reset. It fixes the layout of the world, the agent's random choices
        # and the random numbers of anything an adapter computes.
        self._seeds = iter(())
        unseeded_reset = simulator.reset

        def seeded_reset(**_):
            seed = next(self._seeds)
            self.agent.rng = np.random.default_rng(seed)
            self.frozen.reseed(seed)
            return unseeded_reset(seed=seed)

        simulator.reset = seeded_reset

        # With `watch` set, what the simulator reports at each of its steps is kept for
        # the episode being played. It is hooked in the same way as the seed.
        self.watch = False
        self._watched = []
        unwatched_reset, unwatched_step = simulator.reset, simulator.step

        def looked_at(info):
            task = simulator.unwrapped.task
            place = np.asarray(task.agent.pos[:2], np.float64)
            row = {name: float(value) for name, value in info.items()
                   if name.startswith('cost_') and name != 'cost_sum'}
            row.update({'speed': float(np.linalg.norm(task.agent.vel[:2])), 'x': float(place[0]), 'y': float(place[1]),
                        'goals': float(bool(info.get('goal_met', False)))})
            for group in task._obstacles:
                positions = np.asarray(group.pos, np.float64).reshape(-1, 3)[:, :2]
                row[f'near_{group.name}'] = float(np.linalg.norm(positions - place, axis=1).min())
            return row

        def watched_reset(**options):
            result = unwatched_reset(**options)
            # The first row is the world as the reset left it.
            self._watched = [looked_at({})] if self.watch else []
            return result

        def watched_step(action):
            result = unwatched_step(action)
            if self.watch:
                self._watched.append(looked_at(result[-1]))
            return result

        simulator.reset, simulator.step = watched_reset, watched_step

    def run(self, policy, seeds):
        """Plays one episode per seed with `policy`, a function of the form SafeDreamer's own loop takes.

        Returns what that loop recorded of each episode. This is how evaluate.py plays, in
        one go for all the seeds; play() below is built on it.
        """
        seeds = list(seeds)
        self._seeds = iter(seeds)
        episodes = []
        driver = self._driver(self._env)
        driver.on_episode(lambda episode, _worker: episodes.append(episode))
        driver(policy, episodes=len(seeds), lag=self._penalty)
        return episodes

    def replay(self, seed, actions, cameras=False):
        """Steps the simulator through `actions` from the start of episode `seed`, with no agent.

        Returns the reward and the cost that came after each action. Entry t of what comes
        back is what an episode from play() holds at entry t + 1.

        The actions pass through the same wrappers as when the agent plays, so the
        simulator is handed exactly what it was handed then. The cameras are not drawn
        unless `cameras` is set: the world and its physics follow from the seed and the
        actions alone, and leaving the drawing out is several times faster.
        """
        self._seeds = iter([seed])
        drawn = self._simulator_wrapper._render
        self._simulator_wrapper._render = cameras

        def step(action, reset):
            return self._env.step({'action': action[None], 'reset': np.array([reset])})

        try:
            actions = np.asarray(actions, np.float32)
            step(np.zeros_like(actions[0]), True)    # as SafeDreamer's loop starts an episode
            rewards, costs = [], []
            for action in actions:
                came = step(action, False)
                rewards.append(came['reward'][0])
                costs.append(came['cost'][0])
                if came['is_last'][0]:
                    break
        finally:
            self._simulator_wrapper._render = drawn
        return np.array(rewards, np.float32), np.array(costs, np.float32)

    def play(self, seeds, adapter=None):
        """Plays one episode per seed, with `adapter` if given, and hands each over as soon as it ends.

        An adapter is called as adapter(state, proposed) and returns the action to take; see
        FrozenDreamer.policy. Each episode comes back as the arrays described at the top of
        this file.
        """
        for seed in seeds:
            seen, proposals = [], []

            def recording(state, proposed):
                seen.append(state)
                proposals.append(proposed)
                return proposed if adapter is None else adapter(state, proposed)

            started = time.perf_counter()
            played, = self.run(lambda *inputs: self.frozen.policy(*inputs, adapter=recording), [seed])
            written_out = np.concatenate([state['stoch'] for state in seen])
            episode = {
                'seed': seed,
                'deter': np.concatenate([state['deter'] for state in seen]),
                # Each choice is saved as its number, which takes 48 bytes where the zeros and a one take 4,608.
                'stoch': written_out.argmax(-1).astype(np.uint8),
                'classes': written_out.shape[-1],
                'proposed': np.concatenate(proposals),
                'action': played['action'],
                'reward': played['reward'],
                'cost': played['cost'],
                'seconds': round(time.perf_counter() - started, 1),
            }
            if not np.array_equal(states(episode)['stoch'], written_out):
                raise SystemExit('a categorical choice was not exactly zeros and a one, so saving its number loses something')
            if self.watch:
                episode.update(self._by_decision(len(episode['reward'])))
            yield episode

    def _by_decision(self, entries):
        """What was watched at each simulator step of the episode just played, as one entry per decision."""
        at_reset, rows, each = self._watched[0], self._watched[1:], self.steps_per_decision
        if len(rows) != each * (entries - 1):
            raise SystemExit(f'{len(rows)} simulator steps were watched in an episode of {entries - 1} decisions')
        watched = {}
        for name in sorted({name for row in rows for name in row}):
            steps = np.array([row.get(name, 0.0) for row in rows], np.float32).reshape(entries - 1, each)
            if name.startswith('cost_') or name == 'goals':
                came = steps.sum(1)
            elif name == 'speed':
                came = steps.mean(1)
            else:
                came = steps[:, -1]
            watched[name] = np.concatenate([[at_reset.get(name, 0.0)], came]).astype(np.float32)
        return watched


def file_name(seed):
    return f'seed_{seed:04d}.npz'


def save(episode, folder):
    """Saves an episode as FOLDER/seed_NNNN.npz."""
    # Written under another name first, so that a file with the final name is always complete.
    unfinished = folder / f'unfinished_{file_name(episode["seed"])}'
    np.savez_compressed(unfinished, **episode)
    unfinished.rename(folder / file_name(episode['seed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=25)
    parser.add_argument('--first-seed', type=int, default=1000,
                        help='episode k uses seed first-seed + k; the evaluation seeds are 0 to 49')
    parser.add_argument('--out', help='folder for the episodes (default: episodes/ in the run folder)')
    args = parser.parse_args()

    folder = pathlib.Path(args.out or pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'episodes')
    make_folder(folder)
    # Episodes already saved are not played again, so a restarted job carries on.
    seeds = [seed for seed in range(args.first_seed, args.first_seed + args.episodes)
             if not (folder / file_name(seed)).exists()]

    player = Player(args.checkpoint)
    (folder / 'run.json').write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'task': player.task,
        'simulator_steps_per_decision': player.steps_per_decision,
        'rendering': os.environ.get('MUJOCO_GL'),
        'adapter': None,
    }) + '\n')
    for episode in player.play(seeds):
        save(episode, folder)
        print(f'seed {episode["seed"]}: reward {episode["reward"].sum():.2f}, cost {episode["cost"].sum():.0f}, '
              f'{len(episode["reward"])} entries, {episode["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()
