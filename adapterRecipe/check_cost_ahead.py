"""Checks the cost predictor on states the world model rolled forward, which is how the recipe will use it.

The predictor learned from states the model formed on seeing a camera image. To look
ahead, the recipe rolls the model forward a few decisions and asks the predictor about the
states on the way, and those are the model's guesses with no image. This measures how well
the predictor does there.

It takes the test episodes of cost_predictor.py. For each decision in them, and each
number of decisions k in --ahead, the cost really charged is set against the chance the
predictor gives in three states:

    rolled         the state from k decisions earlier, rolled forward k times along the
                   actions that were really taken. With k = 0 this is the state formed on
                   seeing the image: the predictor's own test, and the most to expect.
    other actions  the same, rolled along the episode's actions in a shuffled order. If
                   this scores as well as `rolled`, what the model expects ahead does not
                   depend on the action, and looking ahead could not tell a safe action
                   from an unsafe one.
    not rolled     the state from k decisions earlier, as it was. This is what is known
                   without looking ahead.

Every k is scored on the same decisions, so only how long ago the model last saw an image
differs between the rows. The columns are those of cost_predictor.py, and two more:

    cost     the expected cost added up over all scored decisions, divided by the real cost
    reward   how closely the model's own reward predictor, asked about the same rolled
             state, follows the real reward (a correlation). It was trained with the
             model, so it shows how much of any loss comes from rolling forward at all.

Before any of that, one thing that must hold exactly. The model works out the continuous
part of its next state from the state and action before it, and only then looks at the
image. So a saved state rolled forward once along its saved action must have the
continuous part of the next saved state. If it does not, states and actions are not lined
up and nothing below would mean anything.

From this folder:

  ../msCli/ms submit -t 30m -n check-cost-ahead -- apptainer exec --nv --bind /datasets/smohan \\
      --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \\
      python3 check_cost_ahead.py --predictor RUN_FOLDER/cost_predictor.npz --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib

import numpy as np

from cost_predictor import auroc, auroc_range, chance, read
from episodes import Player, states

KINDS = ('rolled', 'other actions', 'not rolled')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True,
                        help='a cost_predictor.npz, with the cost_predictor.json that was saved beside it')
    parser.add_argument('--ahead', type=int, nargs='+', default=[1, 2, 3, 5, 10, 15],
                        help='the numbers of decisions to roll forward')
    parser.add_argument('--likely', action='store_true',
                        help='roll to the most likely next state, as look_ahead.py does, not to a random draw')
    args = parser.parse_args()

    # `files` are the episodes the predictor was scored on, which it has never been fitted to.
    predictor, trained, files = read(args.predictor, args.checkpoint)
    if trained['ahead'] != 0:
        raise SystemExit('this needs a predictor of the cost of arriving in a state (--ahead 0)')

    # Player is the one place that knows how to build the agent. Its simulator is not used here.
    frozen = Player(args.checkpoint).frozen
    steps = sorted(set(args.ahead))
    if steps[0] < 1:
        raise SystemExit('--ahead takes numbers of decisions from 1 up; 0, the state as it was, is always scored')
    furthest = steps[-1]

    def roll(state, actions):
        """Rolls every state forward along its own actions: actions[i] holds the action each one takes at step i.

        Returns the state numbers and the predicted reward after each number of steps
        asked for (0 is the state as given), and the continuous part after one step.
        """
        numbers, rewards = {0: frozen.numbers(state)}, {0: frozen.reward(state)}
        for step in range(1, furthest + 1):
            state = frozen.roll_forward(state, actions[step - 1], sample=not args.likely)
            if step == 1:
                continuous = state['deter']
            if step in steps:
                numbers[step], rewards[step] = frozen.numbers(state), frozen.reward(state)
        return numbers, rewards, continuous

    roll = frozen.compiled(roll)
    shuffle = np.random.default_rng(0)
    chances = {(kind, step): [] for kind in KINDS for step in [0] + steps}
    predicted_reward = {step: [] for step in [0] + steps}
    costs, rewards, onsets, episodes, gaps = [], [], [], [], []
    for number, file in enumerate(files):
        saved = np.load(file)
        cost, action = saved['cost'], saved['action']
        entries = len(cost)
        frozen.reseed(int(saved['seed']))

        def following(actions):
            """Row i holds, for every entry, the action taken i decisions after it (zeros past the episode's end)."""
            padded = np.concatenate([actions, np.zeros((furthest,) + actions.shape[1:], actions.dtype)])
            return np.stack([padded[i:i + entries] for i in range(furthest)])

        numbers, reward_ahead, continuous = roll(states(saved), following(action))
        other, _, _ = roll(states(saved), following(shuffle.permutation(action)))
        gaps.append(np.abs(continuous[:-1].astype(np.float32) - saved['deter'][1:]))

        # The decisions that are scored: those with `furthest` entries before them, so that every k can reach them.
        costs.append(cost[furthest:])
        rewards.append(saved['reward'][furthest:])
        onsets.append(cost[furthest - 1:-1] == 0)
        episodes.append(np.full(entries - furthest, number))
        for step in [0] + steps:
            # The entries `step` decisions before the scored ones. Rolled `step` times, they stand for the scored ones.
            before = slice(furthest - step, entries - step)
            chances['rolled', step].append(chance(predictor, numbers[step][before]))
            chances['other actions', step].append(chance(predictor, other[step][before]))
            chances['not rolled', step].append(chance(predictor, numbers[0][before]))
            predicted_reward[step].append(reward_ahead[step][before])

    gap = np.concatenate(gaps)
    print(f'{trained["task"]}, {len(files)} test episodes, rolled to '
          + ('the most likely next state' if args.likely else 'a random draw of the next state'))
    print('One roll forward against the next saved state, '
          f'continuous part: mean gap {gap.mean():.2g}, largest {gap.max():.2g}')
    if gap.mean() > 1e-3:
        raise SystemExit('FAIL: a state rolled forward once is not the next saved state, so states and actions are not lined up')

    cost, reward, onset, episode = (np.concatenate(rows) for rows in (costs, rewards, onsets, episodes))
    costly = cost > 0
    result = {}
    for step in [0] + steps:
        # With no roll there is one state to ask about, so the three kinds are the same.
        result[step] = line = {}
        for kind in (KINDS if step else KINDS[:1]):
            given = np.concatenate(chances[kind, step])
            line[kind] = {'AUROC': auroc(given, costly), 'AUROC, 95% range': auroc_range(given, costly, episode),
                          'AUROC, onsets': auroc(given[onset], costly[onset])}
        given = np.concatenate(chances['rolled', step])
        alarms = given >= 0.5
        line['rolled'].update({
            'found': float((alarms & costly).sum() / costly.sum()),
            'right': float((alarms & costly).sum() / alarms.sum()),
            'cost': float(given.sum() * predictor['mean_cost'] / cost.sum()),
            'reward': float(np.corrcoef(np.concatenate(predicted_reward[step]), reward)[0, 1]),
        })

    print(f'{len(cost)} decisions scored, {costly.mean():.1%} of them costly. Each pair is the AUROC, then the AUROC over onsets.')
    print('ahead  rolled (95% range)              found  right  cost   reward   other actions   not rolled')
    for step, line in result.items():
        rolled = line['rolled']
        low, high = rolled['AUROC, 95% range']
        text = (f'{step:5d}  {rolled["AUROC"]:.3f} ({low:.3f} to {high:.3f}) {rolled["AUROC, onsets"]:.3f}  '
                f'{rolled["found"]:.2f}   {rolled["right"]:.2f}   {rolled["cost"]:.2f}   {rolled["reward"]:.2f}  ')
        for kind in KINDS[1:]:
            if kind in line:
                text += f'   {line[kind]["AUROC"]:.3f} {line[kind]["AUROC, onsets"]:.3f}  '
        print(text, flush=True)

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'check_cost_ahead.json'
    out.write_text(json.dumps({
        'checkpoint': trained['checkpoint'],
        'task': trained['task'],
        'predictor': args.predictor,
        'test episodes': trained['files']['test'],
        'rolled to': 'the most likely next state' if args.likely else 'a random draw',
        'decisions scored': len(cost),
        'costly share': float(costly.mean()),
        'one roll against the next saved state, continuous part': {'mean gap': float(gap.mean()), 'largest gap': float(gap.max())},
        'ahead': {str(step): line for step, line in result.items()},
    }, indent=2) + '\n')
    print('PASS: states and actions are lined up; the rows above are measurements, not a verdict')


if __name__ == '__main__':
    main()
