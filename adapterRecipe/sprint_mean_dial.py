"""Sets the mean table of a task's searches, played at several strengths, beside the searches' own adapters.

It plays nothing, and reads scoreboards fetched to the laptop:

    --individual   the scoreboards of the adapters each search chose (one for each search)
    --mean         the scoreboards of the mean table (mean_table.py), one for each task and strength
    --unadapted    the scoreboards of the unadapted agent

Everything is compared over the seeds that the mean table was played on, which the other
two were played on as well.

    python3 sprint_mean_dial.py --individual ../results/revised/evoWMs-board-revised-no-memory-94639 \
        --mean ../results/sprint/evoWMs-mean-dial-95600 --unadapted ../results/heldout/evoWMs-unadapted-93930

For each task it prints the reward kept (as a share of the unadapted agent's), the cost,
and the share of decisions at which an action number is moved by more than 0.1: for the
searches' own adapters as a mean over them, and for the mean table at each strength. The
dial's pick is the weakest strength whose mean cost is within the limit, and the strongest
if none is. It is picked on the same episodes it is reported on, so its cost reads low.
"""

import argparse
import json
import pathlib

import numpy as np


def short(task):
    return task.replace('safetygym_Safety', '').replace('-v0', '')


def boards(folder):
    for file in sorted(pathlib.Path(folder).glob('task-*/scoreboard.json')):
        yield json.loads(file.read_text())


def over(saved, seeds):
    """Mean reward, cost and share of decisions moved by more than 0.1, over the episodes of `seeds`."""
    rows = [row for row in saved['each episode'] if row['seed'] in seeds]
    if len(rows) != len(seeds):
        raise SystemExit(f'{saved["task"]}: a scoreboard lacks some of the seeds')
    return {name: float(np.mean([row[key] for row in rows]))
            for name, key in (('reward', 'reward'), ('cost', 'cost'), ('moved', 'moved by more than 0.1'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--individual', required=True)
    parser.add_argument('--mean', required=True)
    parser.add_argument('--unadapted', nargs='+', required=True)
    parser.add_argument('--out', help='a JSON file for the numbers')
    args = parser.parse_args()

    mean = {}
    for saved in boards(args.mean):
        mean.setdefault(short(saved['task']), {})[saved['played']['strength']] = saved
    table = {}
    for task, by_strength in sorted(mean.items()):
        seeds = set.intersection(*({row['seed'] for row in saved['each episode']} for saved in by_strength.values()))
        limit = next(iter(by_strength.values()))['cost limit']
        usual = [over(saved, seeds) for root in args.unadapted for saved in boards(root)
                 if short(saved['task']) == task and saved['played']['kind'] == 'none']
        own = [over(saved, seeds) for saved in boards(args.individual) if short(saved['task']) == task]
        if not usual or not own:
            continue
        usual = usual[0]
        row = {'episodes': len(seeds), 'unadapted': usual,
               'own': {'adapters': len(own), 'within the limit': int(sum(one['cost'] <= limit for one in own)),
                       'reward kept': float(np.mean([one['reward'] for one in own]) / usual['reward']),
                       'cost': float(np.mean([one['cost'] for one in own])), 'moved': float(np.mean([one['moved'] for one in own]))}}
        for strength, saved in sorted(by_strength.items()):
            found = over(saved, seeds)
            row[f'mean x {strength:g}'] = {'reward kept': found['reward'] / usual['reward'], 'cost': found['cost'], 'moved': found['moved']}
        within = [strength for strength in sorted(by_strength) if row[f'mean x {strength:g}']['cost'] <= limit]
        row['dial'] = {'strength': within[0] if within else max(by_strength), **row[f'mean x {(within[0] if within else max(by_strength)):g}']}
        table[task] = row

    names = [name for name in next(iter(table.values())) if name.startswith('mean x')]
    print(f'{"task":16s} {"own adapters":>22s}  ' + '  '.join(f'{name:>18s}' for name in names) + '   dial')
    for task, row in table.items():
        own = row['own']
        print(f'{task:16s} {own["reward kept"]:5.0%} at {own["cost"]:5.2f}, {own["within the limit"]} of {own["adapters"]}  '
              + '  '.join(f'{row[name]["reward kept"]:8.0%} at {row[name]["cost"]:5.2f}' for name in names)
              + f'   x {row["dial"]["strength"]:g}')
    limit = 2
    print()
    for name in ['own'] + names + ['dial']:
        kept = np.mean([row[name]['reward kept'] for row in table.values()])
        cost = np.mean([row[name]['cost'] for row in table.values()])
        moved = np.mean([row[name]['moved'] for row in table.values()])
        inside = sum(row[name]['cost'] <= limit for row in table.values())
        print(f'{name:12s} over {len(table)} tasks: reward kept {kept:.0%}, cost {cost:.2f}, mean cost within the limit on {inside}, moved {moved:.0%}')
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(table, indent=2) + '\n')


if __name__ == '__main__':
    main()
