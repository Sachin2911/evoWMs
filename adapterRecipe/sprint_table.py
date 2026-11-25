"""Sets the arms of a short comparison side by side, task by task, from scoreboards fetched to the laptop.

It plays nothing. An arm is a folder of scoreboards (task-NNNN/scoreboard.json and
scoreboard.npz, as scoreboard.py writes them), one for each adapter that was scored. The
adapters of one task in one arm differ only in the seed of the search that found them.

    python3 sprint_table.py --unadapted ../results/evoWMs-round0-none-93421 ../results/heldout/evoWMs-unadapted-93930 \
        --arm present=../results/sprint/board-present --arm tail=../results/sprint/board-tail

For each arm and task it prints, as means over the arm's adapters:

    reward kept   the reward as a share of the unadapted agent's on the same seeds
    cost          the mean cost of an episode, and how many of the adapters are within the limit
    moved         the share of decisions at which an action number was moved by more than 0.1
    stuck         the share of decisions at which the robot is stuck: over the next --span
                  decisions it covers under a tenth of what the unadapted agent usually
                  covers in that many (the median over its decisions), as check_hand_back.py
                  counts it
    speed         the mean speed as a share of the unadapted agent's

The unadapted agent's own row comes first for each task.
"""

import argparse
import json
import pathlib

import numpy as np


def covered(x, y, span):
    """The distance between where the robot is at each decision and where it is `span` decisions later, for every episode."""
    return np.hypot(x[:, span:] - x[:, :-span], y[:, span:] - y[:, :-span])


def short(task):
    return task.replace('safetygym_Safety', '').replace('-v0', '')


def board(folder, seeds=None):
    """What one scoreboard holds, over the episodes whose seeds are in `seeds` (all of them if none are given)."""
    saved = json.loads((folder / 'scoreboard.json').read_text())
    rows = [row for row in saved['each episode'] if seeds is None or row['seed'] in seeds]
    by_decision = np.load(folder / 'scoreboard.npz')
    chosen = np.isin(by_decision['seed'], [row['seed'] for row in rows])
    return {'task': short(saved['task']), 'played': saved['played'], 'limit': saved['cost limit'],
            'seeds': {row['seed'] for row in rows},
            'reward': float(np.mean([row['reward'] for row in rows])),
            'cost': float(np.mean([row['cost'] for row in rows])),
            'moved': float(np.mean([row['moved by more than 0.1'] for row in rows])),
            'speed': float(by_decision['speed'][chosen][:, 1:].mean()),
            'x': by_decision['x'][chosen], 'y': by_decision['y'][chosen]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--unadapted', nargs='+', required=True, help='folders of scoreboards of the unadapted agent (--kind none)')
    parser.add_argument('--arm', action='append', required=True, help='NAME=FOLDER, a folder of scoreboards; repeatable')
    parser.add_argument('--span', type=int, default=20, help='decisions over which being stuck is judged')
    parser.add_argument('--out', help='a JSON file for the numbers')
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, folder = spec.partition('=')
        for file in sorted(pathlib.Path(folder).glob('task-*/scoreboard.json')):
            if (file.parent / 'scoreboard.npz').exists():
                one = board(file.parent)
                arms.setdefault(one['task'], {}).setdefault(name, []).append(one)

    table = {}
    for root in args.unadapted:
        for file in sorted(pathlib.Path(root).glob('task-*/scoreboard.json')):
            saved = json.loads(file.read_text())
            task = short(saved['task'])
            if task not in arms or saved['played']['kind'] != 'none' or not (file.parent / 'scoreboard.npz').exists():
                continue
            seeds = set.intersection(*(one['seeds'] for boards in arms[task].values() for one in boards))
            usual = board(file.parent, seeds)
            distance = float(np.median(covered(usual['x'], usual['y'], args.span)))
            table[task] = {'unadapted': {'reward': usual['reward'], 'cost': usual['cost'], 'episodes': len(seeds), 'speed': usual['speed'],
                                         'usual distance': distance}}
            for name, boards in arms[task].items():
                rows = []
                for one in boards:
                    if one['seeds'] != seeds:
                        raise SystemExit(f'{task}, {name}: the arms were not all scored on the same seeds')
                    stuck = float((covered(one['x'], one['y'], args.span) < distance / 10).mean())
                    rows.append({'reward kept': one['reward'] / usual['reward'], 'reward': one['reward'], 'cost': one['cost'],
                                 'moved': one['moved'], 'stuck': stuck, 'speed': one['speed'] / usual['speed']})
                table[task][name] = {
                    'adapters': len(rows), 'within the limit': int(sum(row['cost'] <= boards[0]['limit'] for row in rows)),
                    **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
                    'each': rows}

    for task, by_arm in table.items():
        usual = by_arm['unadapted']
        print(f'\n{task}: unadapted reward {usual["reward"]:.2f}, cost {usual["cost"]:.1f} over {usual["episodes"]} episodes')
        print(f'{"arm":28s} {"reward kept":>11s} {"cost":>6s} {"within":>7s} {"moved":>6s} {"stuck":>6s} {"speed":>6s}')
        for name, row in by_arm.items():
            if name == 'unadapted':
                continue
            print(f'{name:28s} {row["reward kept"]:11.0%} {row["cost"]:6.2f} {row["within the limit"]:>3d} of {row["adapters"]:<2d} '
                  f'{row["moved"]:6.0%} {row["stuck"]:6.0%} {row["speed"]:6.0%}'
                  + '   each: ' + ', '.join(f'{one["reward kept"]:.0%} at {one["cost"]:.1f}' for one in row['each']))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(table, indent=2) + '\n')


if __name__ == '__main__':
    main()
