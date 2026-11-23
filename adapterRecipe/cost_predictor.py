"""Trains the small cost predictor on saved episodes and scores it on episodes kept apart.

The frozen agent was trained on reward alone, so nothing in it predicts cost. This is the
one piece the recipe adds (brainStorming/adapterRecipeDesign.md, decision 1): from the
numbers of an internal state, the chance that a cost is charged. It can answer one of two
questions:

    --ahead 0   Was a cost charged for arriving in this state? This is the counterpart of
                the agent's reward predictor and the one the recipe uses. It is about the
                state alone, so it stays true when an adapter changes what the agent does
                next. What lies ahead is then found by rolling the world model forward and
                asking about each state on the way.
    --ahead H   Will a cost be charged within the next H decisions? This is about the state
                and about how the unadapted agent carries on from it, so an adapter that
                changes the agent's behaviour changes the right answer.

The predictor is a logistic regression, one weight per state number and an offset:

    chance = 1 / (1 + exp(-(numbers . weights + offset)))

The expected cost is the chance times the mean cost of a costly decision in the training
episodes.

Episodes are split whole, in seed order: the first --test files are the test episodes, the
next --validation choose how strongly large weights are held back, and the rest train. The
test episodes are used once, at the end, for these numbers:

    AUROC          the share of (costly, free) pairs of decisions in which the costly one
                   got the higher chance. 0.5 is guessing and 1 is perfect. Its 95% range
                   comes from redrawing the test episodes with replacement.
    AUROC, onsets  the same over only the decisions before which no cost was being
                   charged. A robot stays in a hazard for several decisions, so most costly
                   decisions follow a costly one. This leaves those out.
    found, right   of the costly decisions, the share given a chance of a half or more,
                   and of the decisions given a half or more, the share that were costly.
    episode cost   with --ahead 0, the expected cost added up over each test episode,
                   against the cost the episode really had.
    control        the AUROC after training on the same states with the answers shuffled.
                   It should be about 0.5.

From this folder:

  ../msCli/ms submit -t 1h -n cost-predictor -- apptainer exec --bind /datasets/smohan \\
      /home-mscluster/smohan/images/safeDreamer.sif python3 cost_predictor.py \\
      --episodes /datasets/smohan/adapterRecipe/episodes/PointGoal1
"""

import argparse
import json
import os
import pathlib

import numpy as np
from scipy import optimize, special, stats

from episodes import make_folder, states

# How strongly large weights are held back. Each is tried, and the one that fits the
# validation episodes best is kept.
STRENGTHS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def windows(cost, ahead):
    """For an episode's costs: the entries that can be used, the cost in each one's window, and the cost just before it.

    With ahead = 0 the window of entry t is entry t itself. With ahead = H it is the H
    entries after t.
    """
    if ahead == 0:
        # The entry at reset is left out: no decision led to it, and its cost of 0 was not measured.
        return np.arange(1, len(cost)), cost[1:], cost[:-1]
    # in_window[t] = cost[t + 1] + ... + cost[t + ahead], for every t that has all of them.
    in_window = np.lib.stride_tricks.sliding_window_view(cost[1:], ahead).sum(-1)
    return np.arange(len(in_window)), in_window, cost[:len(in_window)]


def load(files, ahead):
    """Reads episodes as rows, one per usable entry.

    numbers   the internal state, written as FrozenDreamer.numbers writes it
    amount    the cost in the row's window, and `costly` whether there was any
    charged   whether a cost was being charged just before the window
    episode   which of `files` the row came from
    """
    numbers, amount, before, episode = [], [], [], []
    for number, file in enumerate(files):
        saved = np.load(file)
        entries, in_window, just_before = windows(saved['cost'], ahead)
        if 'z' in saved:
            # An episode of TD-MPC2: its latent state is one row of numbers already.
            numbers.append(saved['z'][entries].astype(np.float32))
        else:
            state = states(saved, entries)
            numbers.append(np.concatenate(
                [state['deter'], state['stoch'].reshape(len(entries), -1)], -1).astype(np.float32))
        amount.append(in_window)
        before.append(just_before)
        episode.append(np.full(len(entries), number))
    amount = np.concatenate(amount)
    return {'numbers': np.concatenate(numbers), 'amount': amount, 'costly': amount > 0,
            'charged': np.concatenate(before) > 0, 'episode': np.concatenate(episode)}


def score(predictor, numbers):
    """The number that chance() squashes into a chance. Higher means a cost is more likely."""
    return (numbers @ predictor['weights']).astype(np.float64) + predictor['offset']


def chance(predictor, numbers):
    """The chance of a cost, for each row of state numbers."""
    return special.expit(score(predictor, numbers))


def expected_cost(predictor, numbers):
    """chance() times the mean cost of a costly decision, written for the GPU.

    It is for use inside FrozenDreamer.compiled, where numpy and scipy cannot go.
    `predictor` then holds the weights, offset and mean_cost.
    """
    import jax
    return jax.nn.sigmoid(numbers @ predictor['weights'] + predictor['offset']) * predictor['mean_cost']


def read(file, checkpoint):
    """Reads a predictor this script saved, for use with the agent in `checkpoint`.

    Returns its arrays, what was recorded when it was trained, and the files of its test
    episodes, which it was never fitted to.
    """
    file = pathlib.Path(file)
    predictor = dict(np.load(file))
    trained = json.loads(file.with_suffix('.json').read_text())
    if trained['checkpoint'] != pathlib.Path(checkpoint).name:
        raise SystemExit(f'the predictor was trained on the states of {trained["checkpoint"]}')
    return predictor, trained, files_of(trained, 'test')


def files_of(trained, part):
    """The episode files that were one part of a predictor's episodes: 'test', 'validation' or 'training'.

    `trained` is what was recorded when the predictor was trained, as read() returns it.
    """
    # The parts lie one after another in seed order, as main() split them.
    files = sorted(pathlib.Path(trained['episodes']).glob('seed_*.npz'))
    for name in ('test', 'validation', 'training'):
        first, last, count = trained['files'][name]
        chosen, files = files[:count], files[count:]
        if name == part:
            break
    if not chosen or [chosen[0].name, chosen[-1].name] != [first, last]:
        raise SystemExit(f'the {part} episodes of the predictor are no longer in {trained["episodes"]}')
    return chosen


def misfit(scores, costly):
    """How badly the scores fit the answers: the mean of -log(chance given to what happened). 0 is perfect."""
    return np.mean(np.logaddexp(0, scores) - costly * scores)


def fit(numbers, costly, strength, start=None, steps=1000):
    """Finds the weights and offset with the smallest misfit plus `strength` times the sum of squared weights.

    The search for them begins at the predictor `start` if one is given, and takes at
    most `steps` steps. Returns the predictor and how the search for it ended.
    """
    def penalised(parameters):
        weights, offset = parameters[:-1], parameters[-1]
        scores = score({'weights': weights.astype(np.float32), 'offset': offset}, numbers)
        value = misfit(scores, costly) + strength * weights @ weights
        # How the value changes with each weight and with the offset, which the search follows downhill.
        error = (special.expit(scores) - costly) / len(costly)
        slope = (error.astype(np.float32) @ numbers).astype(np.float64) + 2 * strength * weights
        return value, np.append(slope, error.sum())

    if start is None:
        share = costly.mean()
        begin = np.zeros(numbers.shape[1] + 1)
        begin[-1] = np.log(share / (1 - share))    # with all weights zero, this offset gives every row the overall share
    else:
        begin = np.append(start['weights'], start['offset']).astype(np.float64)
    found = optimize.minimize(penalised, begin, jac=True, method='L-BFGS-B', options={'maxiter': steps})
    predictor = {'weights': found.x[:-1].astype(np.float32), 'offset': np.float32(found.x[-1])}
    # A search that stops before it has settled still gives a usable predictor, but it is said.
    reason = found.message.decode() if isinstance(found.message, bytes) else str(found.message)
    return predictor, {'steps': int(found.nit), 'settled': bool(found.success), 'ended': reason}


def auroc(chances, costly):
    """The share of (costly, free) pairs of rows in which the costly row has the higher chance."""
    costly_rows = int(costly.sum())
    free_rows = len(costly) - costly_rows
    if not costly_rows or not free_rows:
        return float('nan')
    ranks = stats.rankdata(chances)    # rows with equal chances share a rank, so a tie counts as half
    return float((ranks[costly].sum() - costly_rows * (costly_rows + 1) / 2) / (costly_rows * free_rows))


def auroc_range(chances, costly, episode, draws=1000):
    """The range that holds 95% of the AUROCs when the episodes are redrawn with replacement."""
    random = np.random.default_rng(0)
    rows_of = [np.flatnonzero(episode == number) for number in np.unique(episode)]
    values = []
    for _ in range(draws):
        rows = np.concatenate([rows_of[pick] for pick in random.integers(len(rows_of), size=len(rows_of))])
        values.append(auroc(chances[rows], costly[rows]))
    return [float(value) for value in np.nanpercentile(values, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--episodes', required=True, help='folder of episodes saved by episodes.py')
    parser.add_argument('--ahead', type=int, default=0,
                        help='0: a cost for arriving in the state. H: a cost within the next H decisions')
    parser.add_argument('--test', type=int, default=40, help='episodes kept for the final scoring')
    parser.add_argument('--validation', type=int, default=20, help='episodes that choose the strength')
    parser.add_argument('--train', type=int, help='training episodes to use (default: all the rest)')
    parser.add_argument('--out', help='folder for the predictor (default: the run folder)')
    args = parser.parse_args()

    folder = pathlib.Path(args.episodes)
    files = sorted(folder.glob('seed_*.npz'))
    held_back = args.test + args.validation
    # The test episodes come first, so collecting more episodes later adds to the training ones only.
    split = {'test': files[:args.test], 'validation': files[args.test:held_back],
             'training': files[held_back:][:args.train]}
    if not all(split.values()):
        raise SystemExit(f'{len(files)} episodes in {folder} are too few for this split')
    test, validation, training = (load(split[name], args.ahead) for name in ('test', 'validation', 'training'))
    if not 0 < training['costly'].mean() < 1:
        raise SystemExit('the training episodes are all costly or all free, so there is nothing to fit')
    mean_cost = float(training['amount'][training['costly']].mean())
    collected = json.loads((folder / 'run.json').read_text())
    print(f'{collected["task"]}, ahead {args.ahead}: ' + ', '.join(f'{len(split[name])} {name}' for name in split)
          + f' episodes. Costly: {training["costly"].mean():.1%} of training rows, mean cost of a costly one {mean_cost:.2f}')

    tried = []
    for strength in STRENGTHS:
        predictor, search = fit(training['numbers'], training['costly'], strength)
        fits = misfit(score(predictor, validation['numbers']), validation['costly'])
        tried.append({'strength': strength, 'validation misfit': float(fits), **search, 'predictor': predictor})
        print(f'strength {strength:g}: validation misfit {fits:.4f}, {search["steps"]} steps'
              + ('' if search['settled'] else f', not settled ({search["ended"]})'), flush=True)
    best = min(tried, key=lambda one: one['validation misfit'])
    predictor = {**best['predictor'], 'mean_cost': np.float32(mean_cost), 'ahead': args.ahead}
    if best['strength'] in (STRENGTHS[0], STRENGTHS[-1]):
        print('the best strength is at the end of the list, so a better one may lie beyond it')

    # The final scoring, on the test episodes.
    chances = chance(predictor, test['numbers'])
    costly, onsets = test['costly'], ~test['charged']
    alarms = chances >= 0.5
    shuffled = np.random.default_rng(0).permutation(training['costly'])
    control, _ = fit(training['numbers'], shuffled, best['strength'])
    result = {
        'costly share of test rows': float(costly.mean()),
        'misfit': float(misfit(score(predictor, test['numbers']), costly)),
        'AUROC': auroc(chances, costly),
        'AUROC, 95% range': auroc_range(chances, costly, test['episode']),
        'AUROC, onsets': auroc(chances[onsets], costly[onsets]),
        'AUROC, onsets, 95% range': auroc_range(chances[onsets], costly[onsets], test['episode'][onsets]),
        'found': float((alarms & costly).sum() / costly.sum()),
        'right': float((alarms & costly).sum() / alarms.sum()),
        'control AUROC': auroc(chance(control, test['numbers']), costly),
    }
    if args.ahead == 0:
        # A row's amount is then the cost of one decision, so an episode's amounts add up to its cost.
        expected = np.bincount(test['episode'], chances * mean_cost)
        real = np.bincount(test['episode'], test['amount'])
        result.update({
            'episode cost, real mean': float(real.mean()),
            'episode cost, expected mean': float(expected.mean()),
            'episode cost, mean gap': float(np.abs(expected - real).mean()),
            'episode cost, correlation': float(np.corrcoef(expected, real)[0, 1]),
        })
    print(f'chosen strength {best["strength"]:g}. On the test episodes:')
    for name, value in result.items():
        print(f'  {name}: ' + (f'{value[0]:.3f} to {value[1]:.3f}' if isinstance(value, list) else f'{value:.3f}'))

    out = pathlib.Path(args.out or os.environ.get('MS_RUN_DIR', '.'))
    make_folder(out)
    np.savez(out / 'cost_predictor.npz', **predictor)
    (out / 'cost_predictor.json').write_text(json.dumps({
        'episodes': str(folder),
        'checkpoint': collected['checkpoint'],
        'task': collected['task'],
        'ahead': args.ahead,
        'files': {name: [chosen[0].name, chosen[-1].name, len(chosen)] for name, chosen in split.items()},
        'costly share of training rows': float(training['costly'].mean()),
        'mean cost of a costly row': mean_cost,
        'tried': [{name: value for name, value in one.items() if name != 'predictor'} for one in tried],
        'chosen strength': best['strength'],
        'test': result,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
