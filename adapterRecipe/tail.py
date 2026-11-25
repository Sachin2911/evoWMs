"""Fits the cost tail: from the numbers of an internal state, the cost the unadapted agent runs up over the decisions after it.

The look-ahead adds the agent's value estimate to every future's reward, for the reward
still to come after the last imagined decision, and until now it added nothing to the
cost. A future in which the robot stops then keeps all its reward to come and shows no
cost, so stopping looks free. The tail is the counterpart of the value estimate for cost:
look_ahead.hold_by_stretch adds it to the cost of every future at each length it is read.

It answers for the state and for how the unadapted agent carries on from it, as the value
estimate does. It is fitted once, to the unadapted agent's episodes, and a search does not
fit it again.

The predictor has the cost predictor's form, so look_ahead.expected() reads it:

    cost to come = mean_cost / (1 + exp(-(numbers . weights + offset)))

Here mean_cost is a ceiling, the largest cost to come among the fitted rows, and the
logistic regression is fitted to the cost to come as a share of that ceiling.

Episodes are split whole, in seed order, as cost_predictor.py splits them: the first
--test files are scored at the end, the next --validation choose how strongly large
weights are held back, and the rest are fitted. The cost predictor of the same folder was
then never fitted to this script's test episodes either.

On the test episodes it prints:

    rank agreement   Spearman's correlation of the predicted and the real cost to come
    R2               the share of the real one's variance that the prediction explains
    mean             the mean predicted against the mean real

From this folder:

  ../msCli/ms submit -t 2h -n tail -- bash on_task.sh tail.py \\
      --episodes /datasets/smohan/adapterRecipe/episodes/TASK \\
      --out /datasets/smohan/adapterRecipe/tails/TASK/tail.npz --task=PointButton1
"""

import argparse
import json
import pathlib

import numpy as np

AHEAD = 40         # decisions the cost to come is added up over: as many as an imagined roll of the search
STRENGTH = 1e-4    # how strongly large weights are held back when fit() is not told; main() chooses one


def to_come(cost, ahead=AHEAD):
    """For every entry of an episode, the cost charged over the `ahead` entries after it.

    The last `ahead` entries have no full window after them, and get NaN.
    """
    cost = np.asarray(cost, np.float64)
    if len(cost) <= ahead:
        return np.full(len(cost), np.nan)
    in_window = np.lib.stride_tricks.sliding_window_view(cost[1:], ahead).sum(-1)
    return np.concatenate([in_window, np.full(ahead, np.nan)])


def fit(numbers, amounts, strength=STRENGTH, ceiling=None, hidden=0, steps=1000):
    """Fits a predictor of `amounts`, the cost to come, from the state numbers of the same rows.

    ceiling    the most the predictor can say (default: the largest of `amounts`). Amounts
               above it are fitted as if they were the ceiling
    hidden     units of one hidden layer between the state and the answer (0: none)

    Returns the predictor in the cost predictor's form: weights, offset and mean_cost, and
    with a hidden layer first and first_offset. look_ahead.expected() gives its answer.
    """
    from cost_predictor import fit as fit_shares

    amounts = np.asarray(amounts, np.float64)
    ceiling = float(amounts.max() if ceiling is None else ceiling)
    if not ceiling > 0:
        raise SystemExit('no row has a cost to come, so there is nothing to fit')
    # A share between 0 and 1 is fitted as a chance is: the misfit is smallest where the
    # predicted share is the mean share of the rows that look alike.
    shares = np.clip(amounts / ceiling, 0, 1)
    predictor, _ = fit_shares(numbers, shares, strength, steps=steps, hidden=hidden)
    return {**predictor, 'mean_cost': np.float32(ceiling)}


def predicted(predictor, numbers):
    """The cost to come that `predictor` gives each row of state numbers."""
    from look_ahead import expected
    return np.asarray(expected(np, predictor, np.asarray(numbers, np.float32)), np.float64)


def read(file, checkpoint=None):
    """Reads a tail this script saved, as the arrays look_ahead.expected() takes, and the decisions it looks ahead.

    With `checkpoint` it stops if the tail was fitted to the states of another agent.
    """
    file = pathlib.Path(file)
    saved = dict(np.load(file))
    fitted = file.with_suffix('.json')
    if checkpoint is not None and fitted.exists():
        fitted_to = json.loads(fitted.read_text())['checkpoint']
        if fitted_to != pathlib.Path(checkpoint).name:
            raise SystemExit(f'the tail was fitted to the states of {fitted_to}')
    return saved, int(saved.pop('ahead'))


def agreement(guessed, real):
    """How well predicted costs to come match the real ones: rank agreement, R2 and the two means."""
    from scipy import stats

    spread = float(np.sum((real - real.mean()) ** 2))
    return {'rank agreement': float(stats.spearmanr(guessed, real)[0]),
            'R2': float(1 - np.sum((real - guessed) ** 2) / spread) if spread else float('nan'),
            'mean predicted': float(guessed.mean()), 'mean real': float(real.mean()),
            'share of rows with a cost to come': float((real > 0).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True, help='of the agent that played the episodes')
    parser.add_argument('--episodes', required=True, help='folder of the unadapted agent\'s episodes, saved by episodes.py')
    parser.add_argument('--ahead', type=int, default=AHEAD, help='decisions the cost to come is added up over')
    parser.add_argument('--out', required=True, help='the file to write, e.g. tail.npz')
    parser.add_argument('--test', type=int, default=40, help='episodes kept for the final scoring')
    parser.add_argument('--validation', type=int, default=20, help='episodes that choose the strength')
    parser.add_argument('--hidden', type=int, default=0, help='units of one hidden layer between the state and the answer (0: none)')
    args = parser.parse_args()

    from cost_predictor import STRENGTHS, load, misfit, score
    from episodes import make_folder

    folder = pathlib.Path(args.episodes)
    collected = json.loads((folder / 'run.json').read_text())
    if collected['checkpoint'] != pathlib.Path(args.checkpoint).name:
        raise SystemExit(f'the episodes in {folder} were played by {collected["checkpoint"]}')
    files = sorted(folder.glob('seed_*.npz'))
    held_back = args.test + args.validation
    split = {'test': files[:args.test], 'validation': files[args.test:held_back], 'training': files[held_back:]}
    if not all(split.values()):
        raise SystemExit(f'{len(files)} episodes in {folder} are too few for this split')
    # load() gives one row per entry that has `ahead` entries after it, with the cost over those as its amount.
    test, validation, training = (load(split[name], args.ahead) for name in ('test', 'validation', 'training'))
    ceiling = float(training['amount'].max())
    print(f'{collected["task"]}, the cost over the next {args.ahead} decisions: '
          + ', '.join(f'{len(split[name])} {name}' for name in split)
          + f' episodes, {len(training["amount"]):,} training rows. A cost to come in {training["costly"].mean():.1%} of them, '
          f'mean {training["amount"].mean():.3f}, largest {ceiling:.1f}', flush=True)

    tried = []
    shares = np.clip(validation['amount'] / ceiling, 0, 1)
    for strength in STRENGTHS:
        predictor = fit(training['numbers'], training['amount'], strength, ceiling, args.hidden)
        fits = float(misfit(score(predictor, validation['numbers']), shares))
        tried.append({'strength': strength, 'validation misfit': fits, 'predictor': predictor})
        print(f'strength {strength:g}: validation misfit {fits:.5f}', flush=True)
    best = min(tried, key=lambda one: one['validation misfit'])
    predictor = best['predictor']
    if best['strength'] in (STRENGTHS[0], STRENGTHS[-1]):
        print('the best strength is at the end of the list, so a better one may lie beyond it')

    result = {name: agreement(predicted(predictor, part['numbers']), part['amount'])
              for name, part in (('test', test), ('training', training))}
    print(f'chosen strength {best["strength"]:g}.')
    for name, found in result.items():
        print(f'  on the {name} episodes: rank agreement {found["rank agreement"]:.3f}, R2 {found["R2"]:.3f}, '
              f'mean predicted {found["mean predicted"]:.3f} against mean real {found["mean real"]:.3f}')

    out = pathlib.Path(args.out)
    make_folder(out.parent)
    np.savez(out, **predictor, ahead=args.ahead)
    out.with_suffix('.json').write_text(json.dumps({
        'episodes': str(folder),
        'checkpoint': collected['checkpoint'],
        'task': collected['task'],
        'ahead': args.ahead,
        'hidden units': args.hidden,
        'files': {name: [chosen[0].name, chosen[-1].name, len(chosen)] for name, chosen in split.items()},
        'ceiling': ceiling,
        'tried': [{name: value for name, value in one.items() if name != 'predictor'} for one in tried],
        'chosen strength': best['strength'],
        **result,
    }, indent=2) + '\n')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
