"""Checks the look-ahead on saved episodes: that it is exact where it must be, and what its numbers say near a cost and away from one.

Three things must hold, or the check fails:

    same again       the same states and actions give exactly the same numbers a second time
    at the limit     where an action number is already at a limit, the two changes for the
                     push that way are exactly 0
    same predictor   the cost predictor written for the GPU gives what cost_predictor.py's
                     own gives for the same states

The rest is measured, for each look-ahead length in --decisions, over the test episodes of
the cost predictor. Nothing here is fitted, and the futures are imagined: no simulator runs.
A decision counts as `charged` if a cost is being charged at it. Of the others it is `near`
if a cost is really charged within the next `decisions` decisions, and `away` if none is
charged within twice as many.

    danger    how well the first number, the expected cost of holding the proposed action,
              tells the decisions with a real cost in the next `decisions` from those
              without (AUROC, as in cost_predictor.py). Beside it, the same for the cost
              predictor asked about the present state only, with no look-ahead.
    quiet     the largest cost change among the pushes, near and away: its middle value and
              the value that 95% of decisions stay under. The design expects these inputs
              to be about zero away from cost, so that the adapter does nothing there.
    way out   over the decisions that are not charged and where holding the proposed action
              is expected to cost half a unit or more: how often some push at least halves
              that, how often no push lowers it by even a tenth, and for each push how
              often it halves it and the reward it gives up when it does. Whether a push
              that looks good to the model is good in the simulator is not tested here.
    sizes     the mean and spread of each of the numbers, which an adapter needs to put
              them on one scale.
    time      milliseconds for one state, as when playing.

From this folder:

  ../msCli/ms submit -t 30m -n check-look-ahead -- apptainer exec --nv --bind /datasets/smohan \\
      --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \\
      python3 check_look_ahead.py --predictor RUN_FOLDER/cost_predictor.npz --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np

from cost_predictor import auroc, auroc_range, chance, expected_cost, read
from episodes import Player, states
from look_ahead import look_ahead, names


def middle_and_most(values):
    """The middle value, and the value that 95% of them stay under."""
    return [float(value) for value in np.percentile(values, [50, 95])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True,
                        help='a cost_predictor.npz, with the cost_predictor.json that was saved beside it')
    parser.add_argument('--decisions', type=int, nargs='+', default=[5, 10, 15],
                        help='the look-ahead lengths to try')
    parser.add_argument('--push', type=float, default=1.0,
                        help='how far a push goes towards the limit: 1 is all the way')
    args = parser.parse_args()

    # `files` are the episodes the predictor was scored on, which it has never been fitted to.
    predictor, trained, files = read(args.predictor, args.checkpoint)
    if trained['ahead'] != 0:
        raise SystemExit('the look-ahead needs a predictor of the cost of arriving in a state (--ahead 0)')
    for_gpu = {name: predictor[name] for name in ('weights', 'offset', 'mean_cost')}
    # Player is the one place that knows how to build the agent. Its simulator is not used here.
    frozen = Player(args.checkpoint).frozen
    lengths = sorted(set(args.decisions))
    longest = lengths[-1]

    def both(state, predictor):
        """The state numbers, and the expected cost the GPU's form of the predictor gives for them."""
        numbers = frozen.numbers(state)
        return numbers, expected_cost(predictor, numbers)

    def compiled_look(decisions):
        def look(state, proposed, predictor):
            return look_ahead(frozen, predictor, state, proposed, decisions, args.push)
        return frozen.compiled(look)

    both = frozen.compiled(both)
    looks = {decisions: compiled_look(decisions) for decisions in lengths}

    same_again, wrong_at_limit, predictor_gap = True, 0, 0.0
    at_limit = 0     # how many (decision, action number, way) are at a limit, over all episodes
    kept = {decisions: {'inputs': [], 'within': [], 'away': []} for decisions in lengths}
    charged, present, episode = [], [], []
    for number, file in enumerate(files):
        saved = np.load(file)
        cost, proposed, state = saved['cost'], saved['proposed'], states(saved)
        action_numbers = proposed.shape[1]
        # The entries that are scored: those with twice the longest look-ahead still to come after them.
        scored = np.arange(len(cost) - 2 * longest)

        numbers, on_gpu = both(state, for_gpu)
        own = chance(predictor, numbers) * predictor['mean_cost']
        predictor_gap = max(predictor_gap, float(np.abs(on_gpu - own).max()))
        charged.append(cost[scored] > 0)
        present.append(own[scored])
        episode.append(np.full(len(scored), number))

        before = np.concatenate([[0], np.cumsum(cost)])    # before[t] is the cost of all entries before t
        held = np.clip(proposed, -1, 1)
        for decisions in lengths:
            inputs = looks[decisions](state, proposed, for_gpu)
            same_again &= np.array_equal(inputs, looks[decisions](state, proposed, for_gpu))
            for action_number in range(action_numbers):
                for way, limit in enumerate((1.0, -1.0)):
                    there = held[:, action_number] == limit
                    # The cost change and the reward change of this push; see look_ahead.names.
                    columns = [1 + 2 * action_number + way, 1 + 2 * action_numbers + 2 * action_number + way]
                    wrong_at_limit += int((inputs[there][:, columns] != 0).sum())
                    at_limit += int(there.sum())
            kept[decisions]['inputs'].append(inputs[scored])
            # The real cost of the `decisions` entries after each scored one, and whether twice as many are free.
            kept[decisions]['within'].append(before[scored + 1 + decisions] - before[scored + 1])
            kept[decisions]['away'].append(before[scored + 1 + 2 * decisions] == before[scored + 1])

    # How long one call takes for a single state, as when playing. The first call for a new shape also compiles.
    milliseconds = {}
    for decisions in lengths:
        one = [({name: part[entry:entry + 1] for name, part in state.items()}, proposed[entry:entry + 1])
               for entry in range(101)]
        looks[decisions](*one[0], for_gpu)
        started = time.perf_counter()
        for single_state, single_action in one[1:]:
            looks[decisions](single_state, single_action, for_gpu)
        milliseconds[decisions] = (time.perf_counter() - started) * 1000 / 100

    charged, present, episode = (np.concatenate(rows) for rows in (charged, present, episode))
    free = ~charged
    called = names(action_numbers)
    pushes = [name.split(', ')[1] for name in called[1:1 + 2 * action_numbers]]
    report = {}
    for decisions in lengths:
        inputs, within, away = (np.concatenate(kept[decisions][name]) for name in ('inputs', 'within', 'away'))
        costly = within > 0
        held_cost = inputs[:, 0]
        cost_changes = inputs[:, 1:1 + 2 * action_numbers]
        reward_changes = inputs[:, 1 + 2 * action_numbers:]
        loudest = np.abs(cost_changes).max(1)
        near, far = free & costly, free & away
        expecting = free & (held_cost >= 0.5)
        # For each of those decisions and each push: does the push at least halve the expected cost?
        halves = cost_changes[expecting] <= -0.5 * held_cost[expecting][:, None]
        lowest = cost_changes[expecting].min(1)
        report[decisions] = {
            'danger': {
                'look-ahead AUROC': auroc(held_cost, costly),
                'look-ahead AUROC, 95% range': auroc_range(held_cost, costly, episode),
                'look-ahead AUROC, not charged': auroc(held_cost[free], costly[free]),
                'present state AUROC': auroc(present, costly),
                'present state AUROC, not charged': auroc(present[free], costly[free]),
                'expected cost over real cost': float(held_cost.sum() / within.sum()),
            },
            'quiet': {
                'near, decisions': int(near.sum()),
                'near, largest cost change': middle_and_most(loudest[near]),
                'away, decisions': int(far.sum()),
                'away, largest cost change': middle_and_most(loudest[far]),
                'away, expected cost of holding': middle_and_most(held_cost[far]),
            },
            'way out': {
                'decisions': int(expecting.sum()),
                'expected cost of holding, middle': float(np.median(held_cost[expecting])),
                'some push at least halves it': float(halves.any(1).mean()),
                'no push lowers it by a tenth': float((lowest > -0.1 * held_cost[expecting]).mean()),
                'pushes': {push: {
                    'at least halves it': float(halves[:, column].mean()),
                    'reward change when it does, middle': float(np.median(
                        reward_changes[expecting][halves[:, column], column])) if halves[:, column].any() else None,
                } for column, push in enumerate(pushes)},
            },
            'sizes': {name: [float(inputs[:, column].mean()), float(inputs[:, column].std())]
                      for column, name in enumerate(called)},
            'milliseconds for one state': milliseconds[decisions],
        }

    print(f'{trained["task"]}, {len(files)} test episodes, {len(charged)} decisions scored, {charged.mean():.1%} of them charged. '
          f'Push {args.push:g}.')
    for decisions, found in report.items():
        danger, quiet, way_out = found['danger'], found['quiet'], found['way out']
        low, high = danger['look-ahead AUROC, 95% range']
        print(f'\nlook-ahead of {decisions} decisions ({found["milliseconds for one state"]:.1f} ms for one state)')
        print(f'  danger: look-ahead AUROC {danger["look-ahead AUROC"]:.3f} ({low:.3f} to {high:.3f}), '
              f'not charged {danger["look-ahead AUROC, not charged"]:.3f}; '
              f'present state {danger["present state AUROC"]:.3f}, not charged {danger["present state AUROC, not charged"]:.3f}; '
              f'expected cost over real {danger["expected cost over real cost"]:.2f}')
        print(f'  quiet: largest cost change, middle and 95%: '
              f'near ({quiet["near, decisions"]} decisions) {quiet["near, largest cost change"][0]:.3f} and {quiet["near, largest cost change"][1]:.3f}; '
              f'away ({quiet["away, decisions"]}) {quiet["away, largest cost change"][0]:.4f} and {quiet["away, largest cost change"][1]:.4f}')
        print(f'  way out: {way_out["decisions"]} decisions expect a cost of 0.5 or more from holding '
              f'(middle {way_out["expected cost of holding, middle"]:.2f}); some push at least halves it at '
              f'{way_out["some push at least halves it"]:.0%}, no push lowers it by a tenth at '
              f'{way_out["no push lowers it by a tenth"]:.0%}')
        for push, does in way_out['pushes'].items():
            gives_up = does['reward change when it does, middle']
            print(f'    {push:14s} at least halves it at {does["at least halves it"]:4.0%}'
                  + ('' if gives_up is None else f'; middle reward change when it does {gives_up:+.3f}'))
        print('  sizes (mean, spread): ' + '; '.join(f'{name} {mean:.3f}, {spread:.3f}' for name, (mean, spread) in found['sizes'].items()))

    verdicts = {
        'same again': bool(same_again),
        'at the limit': wrong_at_limit == 0,
        'same predictor': predictor_gap < 1e-4,
    }
    print(f'\nat a limit: {at_limit} cases, {wrong_at_limit} with a change that is not 0. '
          f'Largest gap between the two forms of the predictor: {predictor_gap:.2g}')
    for name, verdict in verdicts.items():
        print(f'{name}: {verdict}')
    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'check_look_ahead.json'
    out.write_text(json.dumps({
        'checkpoint': trained['checkpoint'],
        'task': trained['task'],
        'predictor': args.predictor,
        'test episodes': trained['files']['test'],
        'push': args.push,
        'decisions scored': len(charged),
        'charged share': float(charged.mean()),
        'verdicts': verdicts,
        'cases at a limit': at_limit,
        'largest gap between the two forms of the predictor': predictor_gap,
        'look-ahead of': {str(decisions): found for decisions, found in report.items()},
    }, indent=2) + '\n')
    if not all(verdicts.values()):
        raise SystemExit('FAIL')
    print('PASS: the three exact checks hold; the rest above is measured, not judged')


if __name__ == '__main__':
    main()
