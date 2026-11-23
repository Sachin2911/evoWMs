"""The frozen TD-MPC2 agent, as the five pieces the adapter recipe is allowed to use, and its Meta-World simulator.

The same five pieces as frozen_agent.py gives for DreamerV3, written from what TD-MPC2 has
(implementation/adapterArchitecture.md, "The two agents"):

    state                    the latent state its encoder gives for an observation: 512 numbers
    step(state, action)      its latent dynamics network, which has no randomness, and its
                             reward network, which reads a state and an action together
    worth(state)             its Q-networks at the action its policy network would take
    propose(state)           its policy network. In real play the agent proposes with its
                             planner, which is too slow to ask at every imagined decision
    numbers(state)           the latent state as it is

Everything here runs in PyTorch, where frozen_agent.py runs in JAX. The adapter with
memory, its look-ahead and the imagined rolls are written with `xp`, the array functions
of whichever agent they are given, so they are the same lines for both.

The cost is ours, as in tdmpc2Probe/collect.py: a sphere the gripper must not enter,
centred halfway between where the gripper starts and the first thing it has to reach. A
decision costs 1 if the gripper is inside the sphere when the decision is made. It is
worked out from the gripper's position in the observation, so the simulator is not
changed and the agent never sees it.

Why it is built this way:

  The value is the mean over all of the agent's Q-networks. TD-MPC2 itself reads two of
  them drawn at random, which would give the same state two values.

  In imagined rolls the proposal is the middle of the policy network's choice, not a draw
  from it. The planner that plays for real ends close to one action, and the policy
  network's draws are spread wider than that, because it is trained to stay spread.

  The adapter is worked out without touching the random numbers the planner draws from,
  so an adapter at zero weights replays the unadapted agent's episode exactly.

  The planner starts each plan from its last one, moved on by a decision, and is not told
  the action that was taken. With an adapter that plan was made for an action the robot
  did not take. It is left that way: the planner then plans again from the state the
  robot is really in.

Run as a script, it plays the agent with no adapter and saves one file per episode, as
episodes.py does for DreamerV3:

  ../msCli/ms submit -t 1h -n episodes-tdmpc2 -- apptainer exec --nv --bind /datasets/smohan \\
      /home-mscluster/smohan/images/tdmpc2.sif python3 frozen_tdmpc2.py --episodes 80 \\
      --out /datasets/smohan/adapterRecipe/episodes/mw-reach \\
      --checkpoint /datasets/smohan/tdmpc2_checkpoints/metaworld/mw-reach-1.pt
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tdmpc2Probe'))
from collect import (GOAL, GRIPPER, NAME_PATTERN, OBJECT, SIMULATOR_STEPS_PER_DECISION, STRAIGHT_TO_GOAL,  # noqa: E402
                     file_name, make_folder, prepare, save, settings)

# The radius of the region the gripper must not enter, in metres, for each task. Each was
# chosen from 40 episodes of the unadapted agent so that it charges a cost at about 7.5
# percent of decisions, as the Safety-Gymnasium tasks do at 5 to 11 percent. It has to
# stay below the smallest distance the gripper starts at, which in mw-window-open is
# 0.11 m: there the radius is the one that charges 5 percent. main() prints the table
# these were read from.
RADIUS = {'mw-reach': 0.07, 'mw-push': 0.065, 'mw-button-press-topdown': 0.13, 'mw-window-open': 0.10}

# The cost limit for these tasks: the highest mean cost of an episode that counts as safe.
# Safety-Gymnasium's limit of 2 is 2 to 4 percent of the unadapted agent's cost there. An
# episode here is 100 decisions of which the unadapted agent spends about 7 in the region,
# so half a decision an episode is about 7 percent of its cost.
COST_LIMIT = 0.5


class FrozenTDMPC2:
    """Wraps a loaded TD-MPC2 agent (a single-task one, so its task number is always None)."""

    names = ('z',)

    def __init__(self, agent):
        import torch
        self._torch = torch
        self._agent, self._model, self._cfg = agent, agent.model, agent.cfg
        self._random = np.random.default_rng(0)

    @property
    def xp(self):
        """The array functions the pieces work with: PyTorch's, which answer to the same names as numpy's."""
        return self._torch

    def as_float(self, array):
        return array.float()

    def _scalar(self, logits):
        """What a reward or Q-network says, as a number. They give a spread over bins, as the agent was trained to."""
        from common import math
        return math.two_hot_inv(logits, self._cfg).squeeze(-1)

    def step(self, state, action, sample=True):
        """One decision in the world model: the state that follows `action`, and the reward predicted for it.

        The model has no randomness, so `sample` changes nothing.
        """
        action = action.clamp(-1, 1)    # the simulator clips an action, and the model has only ever seen clipped ones
        reward = self._scalar(self._model.reward(state['z'], action, None))
        return {'z': self._model.next(state['z'], action, None)}, reward

    def worth(self, state):
        """The agent's estimate of the discounted reward still to come from `state`: its Q-networks at its policy network's action."""
        action = self._model.pi(state['z'], None)[0]
        return self._scalar(self._model.Q(state['z'], action, None, return_type='all')).mean(0)

    def propose(self, state, sample=True):
        """The action the policy network takes in `state`: the middle of its choice."""
        return self._model.pi(state['z'], None)[0]

    def numbers(self, state):
        return state['z']

    def compiled(self, function):
        """Makes `function`, written with the pieces above, callable with plain arrays.

        It takes numpy arrays, or tuples and dicts of them, and returns numpy arrays. Nothing
        is compiled: the name is that of the same thing in frozen_agent.py.
        """
        torch = self._torch
        device = self._agent.device

        def to_device(value):
            if isinstance(value, dict):
                return {name: to_device(part) for name, part in value.items()}
            if isinstance(value, (tuple, list)):
                return tuple(to_device(part) for part in value)
            value = np.asarray(value)
            return torch.as_tensor(value.astype(np.float32) if value.dtype.kind == 'f' else value, device=device)

        def from_device(value):
            if isinstance(value, dict):
                return {name: from_device(part) for name, part in value.items()}
            if isinstance(value, (tuple, list)):
                return tuple(from_device(part) for part in value)
            return value.cpu().numpy()

        def call(*inputs):
            # The planner draws from PyTorch's own random numbers, and so does the policy
            # network even when only the middle of its choice is read. They are put back
            # as they were after the call, so that an adapter at zero weights leaves the
            # planner's draws, and with them the episode, exactly as they would have been.
            on_cpu = torch.random.get_rng_state()
            on_gpu = torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None
            # The random numbers of anything `function` draws follow from reseed(), as in frozen_agent.py.
            torch.manual_seed(int(self._random.integers(2 ** 31 - 1)))
            try:
                with torch.no_grad():
                    return from_device(function(*to_device(inputs)))
            finally:
                torch.random.set_rng_state(on_cpu)
                if on_gpu is not None:
                    torch.cuda.set_rng_state(on_gpu, device)

        return call

    def reseed(self, seed):
        """Fixes the random numbers used by everything made with compiled()."""
        self._random = np.random.default_rng([seed, 1])

    # A saved episode keeps the latent state of every decision as it is.

    def kept(self, episode, entries=slice(None)):
        return {'z': episode['z'][entries]}

    def written_out(self, kept):
        return {'z': kept['z']}

    def rows(self, kept):
        return kept['z'].astype(np.float32)


class Player:
    """A frozen TD-MPC2 agent with its simulator, ready to play seeded episodes of its task, alone or with an adapter."""

    steps_per_decision = SIMULATOR_STEPS_PER_DECISION
    cost_limit = COST_LIMIT

    def __init__(self, checkpoint, radius=None):
        match = NAME_PATTERN.search(str(checkpoint))
        if not match:
            raise SystemExit(f'{checkpoint} is not named like a Meta-World checkpoint, mw-TASK-SEED.pt')
        self.task = match.group(1)
        self.toward = 'goal' if self.task in STRAIGHT_TO_GOAL else 'object'
        self.radius = RADIUS[self.task] if radius is None else radius

        prepare()
        import metaworld.envs  # noqa: F401
        import torch
        from common.seed import set_seed
        from envs import make_env
        from tdmpc2 import TDMPC2

        # The agent and its simulator, built the way TD-MPC2's evaluate.py builds them.
        cfg = settings(self.task, checkpoint)
        self._env = make_env(cfg)
        self.agent = TDMPC2(cfg)
        self.agent.load(str(checkpoint))
        self.frozen = FrozenTDMPC2(self.agent)
        self.action_numbers = int(cfg.action_dim)
        self._set_seed, self._torch = set_seed, torch

    def play(self, seeds, adapter=None, memory=None):
        """Plays one episode per seed, with `adapter` if given, and hands each over as soon as it ends.

        An adapter is called as adapter(state, proposed), or with `memory` as
        adapter(state, proposed, memory), exactly as episodes.Player.play calls one. An
        episode comes back as arrays with one entry per decision, as collect.py describes
        them, with `proposed` for the action the planner chose, `action` for the action
        taken, and `memory` for the memory an adapter read at each decision.
        """
        torch = self._torch
        for seed in seeds:
            started = time.perf_counter()
            # Reset twice, so that the half of the observation that is the scene one step
            # earlier is the same scene standing still (see collect.py).
            for _ in range(2):
                self._set_seed(seed)
                obs = self._env.reset()
            self.frozen.reseed(seed)

            seen, latents, proposals, actions, memories, rewards, successes = [], [], [], [], [], [0.0], [0.0]
            carried, done = memory, False
            while not done:
                with torch.no_grad():
                    latent = self.agent.model.encode(obs.to(self.agent.device).unsqueeze(0), None).cpu().numpy()
                # The agent's own way of acting: its encoder, then its planner, without exploration noise.
                proposed = self.agent.act(obs, t0=len(actions) == 0, eval_mode=True).numpy()[None]
                action = proposed
                if adapter is not None and memory is None:
                    action = adapter({'z': latent}, proposed)
                elif adapter is not None:
                    memories.append(carried)
                    action, carried = adapter({'z': latent}, proposed, carried)
                action = np.asarray(action, np.float32)
                seen.append(obs.numpy())
                latents.append(latent[0])
                proposals.append(proposed[0])
                actions.append(action[0])
                obs, reward, done, info = self._env.step(torch.from_numpy(action[0]))
                rewards.append(float(reward))
                successes.append(float(info['success']))
            # The observation that ends the episode. No action follows it.
            seen.append(obs.numpy())
            with torch.no_grad():
                latents.append(self.agent.model.encode(obs.to(self.agent.device).unsqueeze(0), None).cpu().numpy()[0])
            proposals.append(np.zeros_like(proposals[0]))
            actions.append(np.zeros_like(actions[0]))
            if memories:
                memories.append(carried)

            observations = np.stack(seen)
            gripper = observations[:, GRIPPER]
            start = gripper[0]
            target = observations[0, GOAL if self.toward == 'goal' else OBJECT]
            centre = (start + target) / 2
            distance = np.linalg.norm(gripper - centre, axis=1)
            episode = {
                'seed': seed,
                'z': np.stack(latents),
                'obs': observations,
                'proposed': np.stack(proposals),
                'action': np.stack(actions),
                'reward': np.array(rewards, np.float32),
                'success': np.array(successes, np.float32),
                'gripper': gripper,
                'distance': distance,
                'cost': (distance < self.radius).astype(np.float32),
                'centre': centre,
                'radius': self.radius,
                'start': start,
                'target': target,
                'seconds': round(time.perf_counter() - started, 1),
            }
            if memories:
                episode['memory'] = np.concatenate(memories).astype(np.float32)
            yield episode


def regions(folder):
    """Prints what other radii would have charged, from the distances saved with the episodes in `folder`.

    The agent never sees the cost, so its episodes are the same whatever the radius, and
    the cost of another radius is distance < radius.
    """
    episodes = [np.load(file) for file in sorted(pathlib.Path(folder).glob('seed_*.npz'))]
    distance = np.stack([episode['distance'] for episode in episodes])
    print(f'{len(episodes)} episodes: the gripper starts {distance[:, 0].min():.3f} to {distance[:, 0].max():.3f} m from the centre '
          f'and comes within {np.median(distance.min(1)):.3f} m of it in the median episode')
    for share in (0.05, 0.075, 0.10):
        radius = np.quantile(distance[:, 1:], share)
        inside = distance[:, 1:] < radius
        print(f'  a radius of {radius:.3f} m charges {share:.1%} of decisions: {inside.sum(1).mean():.1f} an episode, '
              f'{inside.any(1).mean():.0%} of episodes enter it, the longest stay is {inside.sum(1).max()} decisions')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=80)
    parser.add_argument('--first-seed', type=int, default=1000)
    parser.add_argument('--radius', type=float, help='radius of the region, in metres (default: the one written down for the task)')
    parser.add_argument('--out', help='folder for the episodes (default: episodes/ in the run folder)')
    args = parser.parse_args()

    folder = pathlib.Path(args.out or pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'episodes')
    make_folder(folder)
    seeds = [seed for seed in range(args.first_seed, args.first_seed + args.episodes)
             if not (folder / file_name(seed)).exists()]
    player = Player(args.checkpoint, args.radius)
    (folder / 'run.json').write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'task': player.task,
        'simulator_steps_per_decision': player.steps_per_decision,
        'region': {'centre_halfway_to': player.toward, 'radius': player.radius},
        'adapter': None,
    }) + '\n')
    for episode in player.play(seeds):
        save(episode, folder)
        print(f'{player.task} seed {episode["seed"]}: reward {episode["reward"].sum():.1f}, success {episode["success"][-1]:.0f}, '
              f'cost {episode["cost"].sum():.0f}, {len(episode["cost"])} entries, {episode["seconds"]} s', flush=True)
    regions(folder)


if __name__ == '__main__':
    main()
