"""Plays seeded episodes of a frozen TD-MPC2 agent on Meta-World and saves them with a cost we define.

The test this feeds: can a small predictor that reads the agent's latent state tell when a
cost is charged? TD-MPC2 was trained on reward alone and Meta-World has no cost, so the
cost is ours: a region the gripper must not enter. It is worked out from the gripper's
position, which is part of every observation. The simulator is not changed, and the agent
never sees the cost.

The region is a sphere. Its centre is halfway between where the gripper starts and the
first thing it has to reach: the object, or the goal in the two tasks with no object to
reach first (mw-reach and mw-reach-wall). A gripper that goes straight there passes
through the middle of the sphere. A decision costs 1 if the gripper is inside the sphere
when the decision is made, and 0 otherwise.

An episode is kept as arrays with one entry per decision, as adapterRecipe/episodes.py
keeps them. Entry t belongs to the decision made on seeing observation t:

    z          the latent state: the 512 numbers the agent's encoder gives for observation
               t. They come in 64 groups of 8; each group is positive and adds up to 1
    obs        observation t itself, 39 numbers. Numbers 0 to 2 are the gripper's position,
               4 to 6 the object's and 36 to 38 the goal's; 18 to 35 are numbers 0 to 17 as
               they were one simulator step earlier
    action     the action the agent's planner chose, 4 numbers between -1 and 1
    reward     what came with observation t: the result of the action of entry t - 1, added
               up over the two simulator steps that one decision lasts
    success    whether the simulator counted the task as done at observation t
    gripper    the gripper's position at observation t: x, y and z in metres. It is the
               position Meta-World puts in the observation, that of the hand, which is
               about 4.5 cm above the point between the two fingers
    distance   from the gripper to the centre of the region
    cost       1 where the distance is below the radius

and once per episode: centre, radius, start (the gripper at reset) and target (the object
or goal that the centre lies halfway to).

The first entry is the observation at reset, where reward and success are 0. The last entry
is the observation that ends the episode: its action is never taken, and is saved as zeros.
An episode is 100 decisions, so 101 entries.

Choosing the radius. The Safety-Gymnasium episodes of adapterRecipe charge a cost at about
5 percent of decisions (PointGoal1) and 11 percent (PointButton1), and the radius should
give 5 to 10 percent. Two things set it:

    how long a crossing lasts    An action moves the point the gripper is pulled towards
                                 by at most 1 cm along each axis per simulator step, so
                                 2 cm per decision. Straight through the centre at that
                                 speed, a sphere of radius r metres is crossed in about
                                 100 * r decisions: 5 of an episode's 100 at r = 0.05, and
                                 10 at r = 0.10. An arm that slows down stays longer, and
                                 one that passes off-centre stays shorter.
    where the gripper starts     It starts as far from the centre as its target is: half
                                 the way between them, about 0.09 to 0.17 m in mw-reach
                                 and mw-push. A radius above that puts the start inside
                                 the region.

So start from the default of 0.05 and look at what a run prints. For each episode it prints
the share of decisions inside the region. At the end it prints, from the distances it
saved, the radius that would have charged 5, 7.5 and 10 percent of the decisions, and the
smallest distance any episode starts at, which the radius must stay below. The agent never
sees the cost, so the radius can be changed after playing: cost = distance < radius gives
exactly the episodes that a new run with that radius would.

Run it inside the image. It takes one or more checkpoints, and plays each on the task in
its file name. From this folder:

  ../msCli/ms submit -t 1h -n collect -- apptainer exec --nv --bind /datasets/smohan \\
      /home-mscluster/smohan/images/tdmpc2.sif python3 collect.py --episodes 200 --checkpoint \\
      /datasets/smohan/tdmpc2_checkpoints/metaworld/mw-push-1.pt
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

# recording.py has the way to make a folder on the cluster.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'safeDreamerBaselines'))
from recording import make_folder  # noqa: E402

# Where TD-MPC2's code is. The image has it at /opt/tdmpc2.
TDMPC2_DIR = os.environ.get('TDMPC2_DIR', '/opt/tdmpc2')

# A checkpoint is named like mw-push-1.pt: the task, and the seed it was trained with.
NAME_PATTERN = re.compile(r'(mw-[a-z-]+)-(\d+)\.pt$')

# Where the positions are in a Meta-World observation.
GRIPPER, OBJECT, GOAL = slice(0, 3), slice(4, 7), slice(36, 39)

# The tasks with no object to reach first: the gripper goes straight to the goal.
STRAIGHT_TO_GOAL = ('mw-reach', 'mw-reach-wall')

# TD-MPC2's Meta-World wrapper repeats each action for this many simulator steps.
SIMULATOR_STEPS_PER_DECISION = 2


def prepare():
    """Makes TD-MPC2's code importable, and clears two things out of the simulator's way.

    Call it before importing anything of TD-MPC2 or Meta-World.
    """
    import contextlib

    import fasteners

    # mujoco-py, which Meta-World runs on, opens a lock file next to its compiled code each
    # time it is imported, in case it has to compile. In the image the code is compiled
    # already and nothing in that folder can be written, so opening the file would fail.
    # The lock is swapped for one that does nothing.
    fasteners.InterProcessLock = lambda path: contextlib.nullcontext()

    # TD-MPC2 tries to import every simulator it supports and carries on without those
    # that fail. Only Meta-World is needed here. A None in this table makes Python refuse
    # the other two at once, so nothing of theirs is loaded.
    for other in ('dm_control', 'mani_skill2'):
        sys.modules[other] = None

    # TD-MPC2 is not an installed package. Its files import each other by short names
    # (common, envs, tdmpc2), which work only with its own folder on the path.
    folder = str(pathlib.Path(TDMPC2_DIR) / 'tdmpc2')
    if folder not in sys.path:
        sys.path.insert(0, folder)


def settings(task, checkpoint=None):
    """TD-MPC2's settings for a single-task agent on `task`, completed the way its own scripts do it."""
    import hydra
    from omegaconf import OmegaConf
    from common.parser import parse_cfg

    cfg = OmegaConf.load(pathlib.Path(TDMPC2_DIR) / 'tdmpc2' / 'config.yaml')
    cfg.pop('defaults')    # an instruction to Hydra, which TD-MPC2's scripts start through and this one does not
    cfg.task = task
    cfg.model_size = 5     # every released single-task agent is the 5-million-weight size
    if checkpoint is not None:
        cfg.checkpoint = str(checkpoint)
    # parse_cfg asks Hydra which folder the script was started in, to name a log folder
    # that is never used here. Hydra is not running, so the question is answered for it.
    hydra.utils.get_original_cwd = os.getcwd
    return parse_cfg(cfg)


def latent(agent, obs):
    """The latent state the agent's encoder gives for one observation."""
    import torch

    with torch.no_grad():
        # The encoder takes a batch, here of one. None says the agent has a single task.
        return agent.model.encode(obs.to(agent.device).unsqueeze(0), None)[0].cpu().numpy()


class Player:
    """A frozen TD-MPC2 agent with its simulator, ready to play seeded episodes of its task."""

    def __init__(self, checkpoint):
        match = NAME_PATTERN.search(str(checkpoint))
        if not match:
            raise SystemExit(f'{checkpoint} is not named like a Meta-World checkpoint, mw-TASK-SEED.pt')
        self.task = match.group(1)
        self.toward = 'goal' if self.task in STRAIGHT_TO_GOAL else 'object'

        prepare()
        # Imported by name first, so that a failure shows its own message. TD-MPC2 would
        # carry on without Meta-World and only say later that it cannot make the task.
        import metaworld.envs  # noqa: F401
        from common.seed import set_seed
        from envs import make_env
        from tdmpc2 import TDMPC2

        # The agent and its simulator, built the way TD-MPC2's evaluate.py builds them.
        cfg = settings(self.task, checkpoint)
        self._env = make_env(cfg)
        self.agent = TDMPC2(cfg)
        self.agent.load(str(checkpoint))    # stops if the checkpoint does not fit the agent the settings built
        self._set_seed = set_seed

    def play(self, seeds, radius):
        """Plays one episode per seed and hands each over as soon as it ends.

        Each episode comes back as the arrays described at the top of this file.
        """
        for seed in seeds:
            started = time.perf_counter()
            # The seed fixes where Meta-World puts the object and the goal at reset, and
            # the candidate actions the planner draws. The reset is done twice because
            # half of an observation is the scene one simulator step earlier, which at a
            # reset is left over from whatever was played before. The second time it is
            # left over from the first reset: the same scene, standing still. An episode
            # then depends on its seed alone.
            for _ in range(2):
                self._set_seed(seed)
                obs = self._env.reset()

            seen, latents, actions, rewards, successes = [], [], [], [0.0], [0.0]
            done = False
            while not done:
                seen.append(obs.numpy())
                latents.append(latent(self.agent, obs))
                # The agent's own way of acting: its encoder, then its planner. eval_mode
                # leaves out the exploration noise, as TD-MPC2 does when it measures an
                # agent during training. The encoder draws no random numbers, so asking
                # it for the latent state above does not change what the planner chooses.
                action = self.agent.act(obs, t0=len(actions) == 0, eval_mode=True)
                obs, reward, done, info = self._env.step(action)
                actions.append(action.numpy())
                rewards.append(float(reward))
                successes.append(float(info['success']))
            # The observation that ends the episode. No action follows it.
            seen.append(obs.numpy())
            latents.append(latent(self.agent, obs))
            actions.append(np.zeros_like(actions[0]))

            observations = np.stack(seen)
            gripper = observations[:, GRIPPER]
            start = gripper[0]
            target = observations[0, GOAL if self.toward == 'goal' else OBJECT]
            centre = (start + target) / 2
            distance = np.linalg.norm(gripper - centre, axis=1)
            yield {
                'seed': seed,
                'z': np.stack(latents),
                'obs': observations,
                'action': np.stack(actions),
                'reward': np.array(rewards, np.float32),
                'success': np.array(successes, np.float32),
                'gripper': gripper,
                'distance': distance,
                'cost': (distance < radius).astype(np.float32),
                'centre': centre,
                'radius': radius,
                'start': start,
                'target': target,
                'seconds': round(time.perf_counter() - started, 1),
            }


def file_name(seed):
    return f'seed_{seed:04d}.npz'


def save(episode, folder):
    """Saves an episode as FOLDER/seed_NNNN.npz."""
    # Written under another name first, so that a file with the final name is always complete.
    unfinished = folder / f'unfinished_{file_name(episode["seed"])}'
    np.savez_compressed(unfinished, **episode)
    unfinished.rename(folder / file_name(episode['seed']))


def summarise(folder):
    """Prints what the episodes saved in a folder say about the region, for choosing its radius."""
    episodes = []
    for file in sorted(folder.glob('seed_*.npz')):
        with np.load(file) as saved:
            episodes.append({name: saved[name] for name in ('distance', 'cost', 'success')})
    if not episodes:
        return
    # The entry at reset is left out of the shares: no decision led to it.
    distances = np.concatenate([episode['distance'][1:] for episode in episodes])
    inside = np.concatenate([episode['cost'][1:] for episode in episodes])
    starts = np.array([episode['distance'][0] for episode in episodes])
    done = np.mean([episode['success'][-1] for episode in episodes])
    never = sum(not episode['cost'].any() for episode in episodes)
    print(f'{folder.name}, {len(episodes)} episodes: the task is done at the end of {done:.0%} of them. '
          f'{inside.mean():.1%} of decisions are inside the region; {never} episodes never enter it')
    print(f'  the gripper starts {starts.min():.3f} to {starts.max():.3f} m from the centre; '
          f'a radius must stay below {starts.min():.3f}')
    for share in (0.05, 0.075, 0.10):
        print(f'  a radius of {np.quantile(distances, share):.3f} m would charge {share:.1%} of these decisions')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True, nargs='+',
                        help='released Meta-World checkpoints, e.g. mw-push-1.pt; each is played on its own task')
    parser.add_argument('--episodes', type=int, default=100, help='episodes per checkpoint')
    parser.add_argument('--first-seed', type=int, default=1000, help='episode k uses seed first-seed + k')
    parser.add_argument('--radius', type=float, default=0.05, help='radius of the region, in metres')
    parser.add_argument('--out', help='folder for the episodes, one sub-folder per task '
                                      '(default: episodes/ in the run folder)')
    args = parser.parse_args()

    out = pathlib.Path(args.out or pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'episodes')
    for checkpoint in args.checkpoint:
        player = Player(checkpoint)
        folder = out / player.task
        make_folder(folder)
        run = {
            'checkpoint': pathlib.Path(checkpoint).name,
            'task': player.task,
            'simulator_steps_per_decision': SIMULATOR_STEPS_PER_DECISION,
            'region': {'centre_halfway_to': player.toward, 'radius': args.radius},
        }
        # Episodes already saved are not played again, so a restarted job carries on. They
        # must then be episodes of this agent and this region.
        described = folder / 'run.json'
        if described.exists() and json.loads(described.read_text()) != run:
            raise SystemExit(f'{folder} holds episodes saved with other settings: {described.read_text().strip()}')
        described.write_text(json.dumps(run) + '\n')
        seeds = [seed for seed in range(args.first_seed, args.first_seed + args.episodes)
                 if not (folder / file_name(seed)).exists()]

        for episode in player.play(seeds, args.radius):
            save(episode, folder)
            print(f'{player.task} seed {episode["seed"]}: reward {episode["reward"].sum():.1f}, '
                  f'success {episode["success"][-1]:.0f}, '
                  f'{episode["cost"][1:].mean():.1%} of decisions inside the region, '
                  f'{len(episode["cost"])} entries, {episode["seconds"]} s', flush=True)
        summarise(folder)


if __name__ == '__main__':
    main()
