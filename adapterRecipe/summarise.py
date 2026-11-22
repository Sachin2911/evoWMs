"""Prints the results of the recipe-search runs side by side.

  python3 summarise.py FOLDER [FOLDER ...] [--csv FILE]

Each FOLDER holds one sub-folder per run, as written by evolve.py through run.sh. Per run:

  - the adapters it kept, played on the 50 evaluation seeds: mean reward, mean cost and
    how many episodes were within the cost limit. "final" is the one picked by fresh
    episodes at the end of the run (runs with --final-centres); the other two are picked
    by the validation episodes alone
  - how many simulator steps the search had used when the centre of the search was first
    within the limit on the validation seeds while keeping at least half the reward
  - how well scores from inside the world model agree with scores from the simulator

and then the same averaged over the runs that differ only in their seed.
"""

import argparse
import collections
import csv
import json
import pathlib

import numpy as np

COST_LIMIT = 2.0


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rank_agreement(first, second):
    """Spearman's rank correlation: 1 if the two put the candidates in the same order, 0 if unrelated."""
    first, second = (np.argsort(np.argsort(values)).astype(float) for values in (first, second))
    if first.std() == 0 or second.std() == 0:
        return float('nan')
    return float(np.corrcoef(first, second)[0, 1])


def read_run(run):
    settings = json.loads((run / 'settings.json').read_text())
    label = (run / 'run.txt').read_text().split()[0] if (run / 'run.txt').exists() else run.name
    validations = read_lines(run / 'validations.jsonl')
    generations = read_lines(run / 'generations.jsonl')
    finished = json.loads((run / 'finished.json').read_text()) if (run / 'finished.json').exists() else None
    row = {'run': label, 'folder': str(run), 'task': settings['task'].split('Safety')[-1].split('-')[0],
           'recipe': label.rsplit('-s', 1)[0], 'finished': finished is not None,
           'unadapted_reward': settings['unadapted_reward'], 'unadapted_cost': settings['unadapted_cost'],
           'weights': settings['weights'], 'validations': validations}
    # The first validation at which the centre was within the limit and still did at least half the job.
    good = [result['steps_used'] for result in validations
            if result['cost'] <= COST_LIMIT and result['reward'] >= 0.5 * settings['unadapted_reward']]
    row['first_good_steps'] = good[0] if good else None
    for name in ('final', 'final_penalty', 'penalty', 'safe_first'):
        episodes = read_lines(run / f'evaluation_{name}.jsonl')
        if episodes and finished and name in finished['kept']:        # a run still evaluating may have the file but no record of it yet
            reward = np.array([episode['reward'] for episode in episodes])
            cost = np.array([episode['cost'] for episode in episodes])
            row[name] = {'reward': reward.mean(), 'reward_sd': reward.std(), 'cost': cost.mean(), 'cost_sd': cost.std(),
                         'within': int((cost <= COST_LIMIT).sum()), 'episodes': len(cost),
                         'steps_used': finished['kept'][name]['steps_used'], 'generation': finished['kept'][name]['generation']}
    # The adapter a run stands for: the one picked by fresh episodes if the run has that stage.
    row['main'] = row.get('final') or row.get('penalty')
    # World model against simulator, two ways: across the candidates of a generation (where
    # candidates played), and across the validations of the centre over the run.
    agreement = {'cost': [], 'reward': []}
    for generation in generations:
        played = [candidate for candidate in generation.get('candidates', []) if candidate['real'] and candidate['imagined']]
        if len(played) > 2:
            for name in agreement:
                agreement[name].append(rank_agreement([candidate['imagined'][name] for candidate in played],
                                                      [candidate['real'][name] for candidate in played]))
    row['agreement_candidates'] = {name: float(np.nanmean(values)) if values else float('nan') for name, values in agreement.items()}
    row['agreement_validations'] = {
        name: rank_agreement([result['imagined'][name] for result in validations], [result[name] for result in validations])
        if len(validations) > 3 and all(result.get('imagined') for result in validations) else float('nan') for name in ('cost', 'reward')}
    row['seconds'] = finished['seconds'] if finished else (validations[-1]['seconds'] if validations else 0)
    row['steps'] = finished['steps'] if finished else (validations[-1]['steps_used'] if validations else 0)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folders', nargs='+', type=pathlib.Path)
    parser.add_argument('--csv', type=pathlib.Path, help='also write one line per run to this file')
    parser.add_argument('--curves', action='store_true', help='also print every validation of every run')
    args = parser.parse_args()
    rows = [read_run(path.parent) for folder in args.folders for path in sorted(folder.glob('*/settings.json'))]
    rows.sort(key=lambda row: (row['task'], row['run']))

    def cell(kept):
        if not kept:
            return f"{'-':>31s}"
        return f"{kept['reward']:6.2f} {kept['cost']:6.2f} {kept['within']:3d}/{kept['episodes']:<3d} {kept['steps_used'] // 1000:5d}k"

    print('Kept adapters on the evaluation seeds: reward, cost, episodes within the limit, thousand steps used when kept')
    print(f"{'task':12s} {'run':52s} {'weights':>7s} | {'kept by fresh episodes (final)':>31s} | {'kept by reward minus cost over limit':>31s} | {'kept safe-first':>31s} | {'first good':>10s} {'steps':>6s} {'min':>4s}")
    for row in rows:
        good = f"{row['first_good_steps'] // 1000}k" if row['first_good_steps'] else 'never'
        print(f"{row['task']:12s} {row['run']:52s} {row['weights']:7d} | {cell(row.get('final'))} | {cell(row.get('penalty'))} | {cell(row.get('safe_first'))} | "
              f"{good:>10s} {row['steps'] // 1000:5d}k {row['seconds'] // 60:4d}{'' if row['finished'] else '  (running)'}")

    print('\nAveraged over seeds (finished runs only): the adapter kept by fresh episodes where the run has one, else by reward minus cost over the limit')
    print(f"{'task':12s} {'recipe':48s} {'runs':>4s} {'reward':>7s} {'cost':>7s} {'within':>7s} {'safe runs':>9s} {'first good':>11s} | {'unadapted':>13s} | agreement cost/reward: candidates, validations")
    groups = collections.defaultdict(list)
    for row in rows:
        if row['finished'] and row['main']:
            groups[row['task'], row['recipe']].append(row)
    for (task, recipe), group in sorted(groups.items(), key=lambda item: (item[0][0], np.mean([row['main']['cost'] for row in item[1]]) > COST_LIMIT, -np.mean([row['main']['reward'] for row in item[1]]))):
        kept = [row['main'] for row in group]
        good = [row['first_good_steps'] for row in group if row['first_good_steps']]
        first_good = f"{np.mean(good) / 1000:.0f}k ({len(good)})" if good else 'never'
        print(f"{task:12s} {recipe:48s} {len(group):4d} {np.mean([k['reward'] for k in kept]):7.2f} {np.mean([k['cost'] for k in kept]):7.2f} "
              f"{np.mean([k['within'] / k['episodes'] for k in kept]):7.0%} {sum(k['cost'] <= COST_LIMIT for k in kept):5d}/{len(kept):<3d} {first_good:>11s} | "
              f"{np.mean([row['unadapted_reward'] for row in group]):5.1f} / {np.mean([row['unadapted_cost'] for row in group]):5.1f} | "
              f"{np.nanmean([row['agreement_candidates']['cost'] for row in group]):5.2f} / {np.nanmean([row['agreement_candidates']['reward'] for row in group]):4.2f}, "
              f"{np.nanmean([row['agreement_validations']['cost'] for row in group]):5.2f} / {np.nanmean([row['agreement_validations']['reward'] for row in group]):4.2f}")

    if args.curves:
        print('\nCentre of the search on the validation seeds: reward / cost at each validation, by thousand steps used')
        for row in rows:
            print(f"{row['task']} {row['run']}")
            print('   ' + '  '.join(f"{result['steps_used'] // 1000}k: {result['reward']:.1f}/{result['cost']:.1f}" for result in row['validations']))

    if args.csv:
        with args.csv.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['task', 'run', 'recipe', 'weights', 'finished', 'unadapted_reward', 'unadapted_cost', 'first_good_steps',
                             'steps', 'seconds', 'kept_rule', 'kept_generation', 'kept_steps_used', 'evaluation_reward',
                             'evaluation_reward_sd', 'evaluation_cost', 'evaluation_cost_sd', 'episodes_within_limit', 'evaluation_episodes'])
            for row in rows:
                for name in ('final', 'final_penalty', 'penalty', 'safe_first'):
                    kept = row.get(name)
                    if kept:
                        writer.writerow([row['task'], row['run'], row['recipe'], row['weights'], row['finished'],
                                         round(row['unadapted_reward'], 3), round(row['unadapted_cost'], 3), row['first_good_steps'],
                                         row['steps'], row['seconds'], name, kept['generation'], kept['steps_used'],
                                         round(kept['reward'], 3), round(kept['reward_sd'], 3), round(kept['cost'], 3),
                                         round(kept['cost_sd'], 3), kept['within'], kept['episodes']])


if __name__ == '__main__':
    main()
