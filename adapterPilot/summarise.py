"""Prints the results of the adapter pilot's evolution runs side by side.

  python3 summarise.py FOLDER

FOLDER holds one sub-folder per run, as written by evolve.py and vast/pilot.sh. For each run:

  - the adapters it kept (after half the step budget and after all of it), played on the 50
    evaluation seeds: mean reward, mean cost, and how many episodes were within the cost limit
  - how the centre of the search did on the validation seeds as steps were used
  - how well scores from inside the world model agree with scores from the simulator
"""

import json
import pathlib
import sys

import numpy as np

COST_LIMIT = 2.0


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rank_agreement(first, second):
    """Spearman's rank correlation: 1 if the two put the candidates in the same order, 0 if unrelated."""
    from scipy.stats import rankdata
    first, second = rankdata(first), rankdata(second)
    if first.std() == 0 or second.std() == 0:
        return float('nan')
    return float(np.corrcoef(first, second)[0, 1])


def main():
    folder = pathlib.Path(sys.argv[1])
    runs = sorted(path.parent for path in folder.glob('*/settings.json'))

    print('Kept adapters on the evaluation seeds 0 to 49')
    print(f"{'run':24s} {'kept after':>11s} {'generation':>10s} {'steps used':>10s} {'reward':>14s} {'cost':>14s} {'within limit':>12s}")
    for run in runs:
        settings = json.loads((run / 'settings.json').read_text())
        for kept in ('adapter_half_budget', 'adapter_full_budget'):
            episodes = read_lines(run / f'evaluation_{kept}.jsonl')
            if not episodes:
                continue
            notes = np.load(run / f'{kept}.npz')
            reward = np.array([episode['reward'] for episode in episodes])
            cost = np.array([episode['cost'] for episode in episodes])
            print(f"{run.name:24s} {kept.split('_')[1] + ' budget':>11s} {int(notes['generation']):10d} {int(notes['steps_used']):10d} "
                  f"{reward.mean():7.2f} ({reward.std():5.2f}) {cost.mean():7.2f} ({cost.std():5.2f}) {int((cost <= COST_LIMIT).sum()):6d} of {len(cost)}")
        if run == runs[-1]:
            print(f"{'unadapted (recorded)':24s} {'':>11s} {'':>10s} {'':>10s} {settings['unadapted_reward']:7.2f} {'':7s} {settings['unadapted_cost']:7.2f}")

    print('\nCentre of the search on the validation seeds: reward / cost at each validation, by thousand steps used')
    for run in runs:
        validations = read_lines(run / 'validations.jsonl')
        under = [result['steps_used'] for result in validations if result['cost'] <= COST_LIMIT]
        print(f"{run.name}: first within the limit after {under[0] if under else 'never'} steps")
        print('   ' + '  '.join(f"{result['steps_used'] // 1000}k: {result['reward']:.1f}/{result['cost']:.1f}" for result in validations))

    print('\nWorld model against simulator: rank agreement between imagined and real scores of the candidates')
    print('that played in the simulator, averaged over generations (1 = same order, 0 = unrelated)')
    print(f"{'run':24s} {'generations':>11s} {'cost':>7s} {'reward':>7s} {'imagined / real cost of the unadapted agent':>44s}")
    for run in runs:
        generations = read_lines(run / 'generations.jsonl')
        settings = json.loads((run / 'settings.json').read_text())
        agreement = {'cost': [], 'reward': []}
        for generation in generations:
            played = [candidate for candidate in generation['candidates'] if candidate['real']]
            for name in agreement:
                agreement[name].append(rank_agreement([candidate['imagined'][name] for candidate in played],
                                                      [candidate['real'][name] for candidate in played]))
        imagined_unadapted = np.mean([generation['imagined_unadapted']['cost'] for generation in generations]) if generations else float('nan')
        print(f"{run.name:24s} {len(generations):11d} {np.nanmean(agreement['cost']) if generations else float('nan'):7.2f} "
              f"{np.nanmean(agreement['reward']) if generations else float('nan'):7.2f} {imagined_unadapted:30.1f} / {settings['unadapted_cost']:.1f}")


if __name__ == '__main__':
    main()
