"""Asks whether a cost can be seen coming in the frozen agent's internal state, for each kind of object that charges one.

On the Button tasks most of the cost an adapter leaves is hits by moving obstacles
(implementation/adapterArchitecture.md, decisions 2 and 14). The cost predictor says
whether arriving in a state is charged, and it gives such a hit a chance of 0.04 one
decision before it happens. Before a predictor of what is coming is built into the
recipe, this measures whether the state holds that at all.

From saved episodes it takes every decision at which nothing is being charged, and asks
of each kind of object: does it charge a cost within the next H decisions? Four answers
are scored, each on episodes it was not fitted to:

    weighted sum   a logistic regression on the state, fitted to this question
    hidden layer   the same with one hidden layer (cost_predictor.py --hidden)
    as now         what the look-ahead has today: a predictor of the cost of arriving,
                   asked about the state rolled forward 1 to H decisions in the world
                   model, and added up. It is rolled along the actions that were really
                   taken, which a look-ahead does not know, so this is the most the
                   present look-ahead could see
    simulator      a small network on two numbers the simulator reports and the agent is
                   never shown: the distance to the nearest object of the kind, and how
                   much that changed since the last decision. It stands for what seeing
                   the nearest object perfectly would give

The kind `any` is a cost from whatever object, which is what the cost predictor is
fitted to. It has no distance, so no simulator answer.

For each answer:

    AUROC    the share of (cost coming, none coming) pairs of decisions in which the one
             with a cost coming got the higher score. 0.5 is guessing and 1 is perfect
    found    of the decisions with a cost coming, the share that are flagged when one
             decision in twenty with none coming is flagged

How to read it. If the state's answers are near the simulator's and far above `as now`,
the state holds what is coming and a predictor fitted to it would give the adapter what
the look-ahead lacks. If the simulator's answer is high and the state's are not, the
frozen agent's state does not hold it, and nothing that reads the state can. If the
simulator's is low too, a cost that far ahead is not settled yet by where things are.

The right answer depends on what is done over the next H decisions, so it belongs to
whoever played the episodes. That is why this is a check and not yet a piece of the recipe.

The episodes are split by seed into --folds parts. Each part in turn is scored, by
predictors that were fitted to the other parts with one of them choosing the strength, as
cost_predictor.py chooses it. Every episode is scored once and never by a predictor that
saw its seed.

An episode that was played without being watched has no cost by kind. The simulator is
then stepped through its saved actions again, with no agent and no camera, and must give
the saved costs exactly.

From this folder, on the unadapted agent's episodes of a task:

  ../msCli/ms submit -t 4h -n check-cost-coming -- bash on_task.sh check_cost_coming.py \\
      --episodes /datasets/smohan/adapterRecipe/episodes/TASK --task=PointButton1
"""

import argparse
import json
import os
import pathlib

import numpy as np

from cost_predictor import STRENGTHS, auroc, auroc_range, fit, misfit, score
from episodes import Player
from look_ahead import expected
from tail import to_come as coming

ANSWERS = ('weighted sum', 'hidden layer', 'as now', 'simulator')


def following(actions, furthest):
    """Row i holds, for every entry of an episode, the action taken i decisions after it (zeros past the episode's end)."""
    padded = np.concatenate([actions, np.zeros((furthest,) + actions.shape[1:], actions.dtype)])
    return np.stack([padded[i:i + len(actions)] for i in range(furthest)])


def chosen(train, answers, validation, validation_answers, hidden=0, steps=1000):
    """The predictor of the strength that fits the validation rows best, as cost_predictor.py chooses it."""
    tried = []
    for strength in STRENGTHS:
        predictor, _ = fit(train, answers, strength, steps=steps, hidden=hidden)
        tried.append((misfit(score(predictor, validation), validation_answers), strength, predictor))
    return min(tried, key=lambda one: one[:2])[2]


def found(scores, answers, flagged=0.05):
    """Of the rows with a cost coming, the share that score above all but `flagged` of the rows with none."""
    return float((scores[answers] > np.quantile(scores[~answers], 1 - flagged)).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', required=True, nargs='+', help='folders of episodes saved by episodes.py or scoreboard.py --keep')
    parser.add_argument('--most', type=int, help='use this many episodes at most, the first in name order (default: all)')
    parser.add_argument('--ahead', type=int, nargs='+', default=[1, 3, 5, 10], help='the numbers of decisions H')
    parser.add_argument('--hidden', type=int, default=32, help='units of the hidden layer')
    parser.add_argument('--steps', type=int, default=300, help='the most steps a fit with a hidden layer takes')
    parser.add_argument('--folds', type=int, default=4, help='parts the seeds are split into; at least 3')
    parser.add_argument('--events', type=int, default=20, help='a kind is left out if its cost begins fewer times than this')
    args = parser.parse_args()

    # Player is the one place that knows how to build the agent. Its simulator is used only to watch again.
    player = Player(args.checkpoint)
    frozen = player.frozen
    files = sorted(file for folder in args.episodes for file in pathlib.Path(folder).glob('seed_*.npz'))[:args.most]
    if not files:
        raise SystemExit(f'no episodes in {args.episodes}')
    episodes, again = [], 0
    for file in files:
        saved = dict(np.load(file))
        if not any(name.startswith('cost_') for name in saved):
            # Not watched when it was played: the simulator is stepped through its actions again.
            _, costs, seen = player.replay_watched(int(saved['seed']), saved['action'][:-1])
            if not np.array_equal(costs, saved['cost'][1:]):
                raise SystemExit(f'{file}: stepping the simulator through the saved actions again did not give the saved costs')
            saved.update(seen)
            again += 1
        episodes.append({'seed': int(saved['seed']), 'cost_any': saved['cost'], 'action': saved['action'],
                         'state': frozen.kept(saved), 'numbers': frozen.rows(frozen.kept(saved)),
                         **{name: values for name, values in saved.items() if name.startswith(('cost_', 'near_'))}})
    kinds = ['any'] + sorted({name[len('cost_'):] for episode in episodes for name in episode if name.startswith('cost_')} - {'any'})

    # One row per entry of every episode. A seed's episodes are all in the same part.
    seeds = sorted({episode['seed'] for episode in episodes})
    part_of = {seed: number % args.folds for number, seed in enumerate(seeds)}
    lengths = [len(episode['cost_any']) for episode in episodes]
    first = np.concatenate([[0], np.cumsum(lengths)])
    numbers = np.concatenate([episode.pop('numbers') for episode in episodes])
    of_episode = np.repeat(np.arange(len(episodes)), lengths)
    entry = np.concatenate([np.arange(length) for length in lengths])
    seed_number = np.array([seeds.index(episode['seed']) for episode in episodes])[of_episode]
    part = np.array([part_of[episode['seed']] for episode in episodes])[of_episode]

    def rows_of(name):
        """What every episode holds under `name`, one row per entry; zeros where an episode does not hold it."""
        return np.concatenate([np.asarray(episode.get(name, np.zeros(length)), np.float64)
                               for episode, length in zip(episodes, lengths)])

    cost_now = rows_of('cost_any')
    task = player.task
    print(f'{task}: {len(episodes)} episodes of {len(seeds)} seeds, {again} of them watched again; '
          f'{len(numbers)} decisions, {(cost_now > 0).mean():.1%} of them charged; kinds: {", ".join(kinds[1:])}', flush=True)

    report = {}
    for ahead in sorted(set(args.ahead)):
        def rolled(state, actions, predictor):
            """The chance of a cost in the state after each of 1 to `ahead` rolls of the world model along `actions`."""
            chances = []
            for step in range(ahead):
                state, _ = frozen.step(state, actions[step], sample=False)
                chances.append(expected(frozen.xp, predictor, frozen.numbers(state)))
            return frozen.xp.stack(chances)

        rolled = frozen.compiled(rolled)
        will_cost = {kind: np.concatenate([coming(episode.get(f'cost_{kind}', np.zeros(length, np.float32)), ahead)
                                           for episode, length in zip(episodes, lengths)]) for kind in kinds}
        # Asked at the decisions at which nothing is being charged and that have `ahead` entries after them.
        asked = (cost_now == 0) & ~np.isnan(will_cost['any'])
        answers = {kind: will_cost[kind] > 0 for kind in kinds}
        begins = {kind: int(sum(((episode.get(f'cost_{kind}', np.zeros(length))[:-1] == 0)
                                 & (episode.get(f'cost_{kind}', np.zeros(length))[1:] > 0)).sum()
                                for episode, length in zip(episodes, lengths))) for kind in kinds}
        # A kind is scored if its cost begins often enough, and every part has decisions with and without it coming.
        scored = [kind for kind in kinds if begins[kind] >= args.events and all(
            0 < answers[kind][asked & (part == number)].sum() < (asked & (part == number)).sum() for number in range(args.folds))]
        scores = {(kind, answer): np.full(len(numbers), np.nan) for kind in scored for answer in ANSWERS}
        distance = {}
        for kind in scored:
            if any(f'near_{kind}' in episode for episode in episodes):
                near = rows_of(f'near_{kind}')
                # How much the distance changed since the last decision; nothing at the first entry of an episode.
                change = np.where(entry > 0, near - np.roll(near, 1), 0.0)
                distance[kind] = np.stack([near, change], 1)

        for number in range(args.folds):
            test, validation = part == number, part == (number + 1) % args.folds
            train = ~test & ~validation

            # As now: a predictor of the cost of arriving, fitted to every entry but the one at reset.
            arrived = entry > 0
            arriving = chosen(numbers[train & arrived], cost_now[train & arrived] > 0,
                              numbers[validation & arrived], cost_now[validation & arrived] > 0)
            for_gpu = {'weights': arriving['weights'], 'offset': arriving['offset'], 'mean_cost': np.float32(1)}
            as_now = np.full(len(numbers), np.nan)
            for which in np.flatnonzero([part_of[episode['seed']] == number for episode in episodes]):
                episode = episodes[which]
                chances = rolled(frozen.written_out(episode['state']), following(episode['action'], ahead), for_gpu)
                as_now[first[which]:first[which + 1]] = chances.sum(0)

            rows = {name: of_part & asked for name, of_part in (('train', train), ('validation', validation), ('test', test))}
            state = {name: numbers[chosen_rows] for name, chosen_rows in rows.items()}
            for kind in scored:
                answer = {name: answers[kind][chosen_rows] for name, chosen_rows in rows.items()}
                weighted = chosen(state['train'], answer['train'], state['validation'], answer['validation'])
                hidden = chosen(state['train'], answer['train'], state['validation'], answer['validation'],
                                hidden=args.hidden, steps=args.steps)
                scores[kind, 'weighted sum'][rows['test']] = score(weighted, state['test'])
                scores[kind, 'hidden layer'][rows['test']] = score(hidden, state['test'])
                scores[kind, 'as now'][rows['test']] = as_now[rows['test']]
                if kind in distance:
                    # Two numbers need no holding back, so the strength is nothing. A few hidden units let
                    # the two work together: near and closing is not the same as near, or as closing.
                    seen, _ = fit(distance[kind][rows['train']], answer['train'], 0.0, steps=args.steps, hidden=8)
                    scores[kind, 'simulator'][rows['test']] = score(seen, distance[kind][rows['test']])
            print(f'  a cost within {ahead}: part {number + 1} of {args.folds} scored', flush=True)

        report[ahead] = {'decisions asked at': int(asked.sum()), 'kinds': {}}
        print(f'\n{task}: a cost within {ahead} decisions, asked at {asked.sum():,} decisions with nothing charged. '
              'Each pair is the AUROC, then the share found.')
        print('kind         begins  coming   ' + ''.join(f'{answer:16s}' for answer in ANSWERS))
        for kind in kinds:
            if kind not in scored:
                print(f'{kind:12s} {begins[kind]:6d}  left out: begins too seldom, or a part has none')
                report[ahead]['kinds'][kind] = {'begins': begins[kind], 'left out': True}
                continue
            is_coming = answers[kind][asked]
            line = {'begins': begins[kind], 'share of decisions with it coming': float(is_coming.mean())}
            text = f'{kind:12s} {begins[kind]:6d}  {is_coming.mean():6.1%}   '
            for answer in ANSWERS:
                given = scores[kind, answer][asked]
                if np.isnan(given).any():
                    text += f'{"":16s}'
                    continue
                line[answer] = {'AUROC': auroc(given, is_coming), 'AUROC, 95% range': auroc_range(given, is_coming, seed_number[asked]),
                                'found': found(given, is_coming)}
                text += f'{line[answer]["AUROC"]:.3f} {line[answer]["found"]:.2f}      '
            report[ahead]['kinds'][kind] = line
            print(text, flush=True)

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'check_cost_coming.json'
    out.write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'task': task,
        'episodes': [str(folder) for folder in args.episodes],
        'episodes scored': len(episodes), 'seeds': len(seeds), 'watched again': again,
        'folds': args.folds, 'hidden units': args.hidden,
        'a cost within': {str(ahead): found_there for ahead, found_there in report.items()},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
