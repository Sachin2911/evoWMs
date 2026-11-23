"""Sets the arms of a wave of searches side by side, with the baselines measured on the same seeds.

It reads run folders that were fetched to the laptop and plays nothing. An arm is every
search of one task with the same settings, differing only in its seed. For each arm it
prints two things:

    as the search chose   the reward and cost of each search's chosen adapter over the
                          real episodes the search itself played with it (final.json)
    on the scoreboard     the same adapter on the development seeds, if scoreboard.py has
                          played it (--kind saved). This is the number to compare, because
                          a search picks the adapter that did best in its own episodes

Both are means over the arm's seeds, with the standard error between seeds, and the share
of seeds whose adapter is within the cost limit. Below the arms come the baselines found in
the same folders: the unadapted agent, steering, and SafeDreamer's own checkpoints
(safeDreamerBaselines/evaluate.py), each with the standard error between episodes. For a
baseline "within" is the share of its episodes within the limit, not of seeds.

A root is a folder that holds the run folders of one machine: a search's scoreboard is
looked for under the same root, by the last two parts of the path the scoreboard recorded
(SWEEP/task-NNNN).

    python3 arms.py ../results/vast/B ../results/vast/C
"""

import argparse
import json
import pathlib

import numpy as np

# The settings of search.py that tell arms apart, with the value that is not worth naming.
NAMED = (('form', 'memory'), ('no_memory', False), ('archive', False), ('begin', 'zero'), ('look', 'likely'),
         ('diagonal', False), ('added_on', False), ('no_stop', False), ('cautious', True), ('damped', True), ('hold', False),
         ('generations', 10), ('checks', 15), ('episodes', 10), ('candidates', 16), ('starts', 256), ('roll', 40),
         ('step', 0.3), ('lowest', 0.25), ('failures', 0.25), ('keep_predictor', False), ('search', 'cma'), ('limit', 2))


def mean_and_error(values):
    values = np.asarray(values, np.float64)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float('nan')


def arm_of(settings):
    """What a search is called: where its settings differ from the usual ones."""
    different = [name.replace('_', ' ') if value is True else f'{name.replace("_", " ")} {value}'
                 for name, usual in NAMED for value in [settings.get(name, usual)] if value != usual]
    return ', '.join(different) or 'one score, with the memory'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('roots', nargs='+', type=pathlib.Path, help='folders of fetched run folders, one per machine')
    parser.add_argument('--json', type=pathlib.Path, help='also write the rows to this file')
    args = parser.parse_args()

    searches, baselines = [], []
    for root in args.roots:
        boards = {}
        for file in sorted(root.rglob('scoreboard.json')):
            saved = json.loads(file.read_text())
            played, each = saved['played'], saved['each episode']
            rewards, costs = [row['reward'] for row in each], [row['cost'] for row in each]
            if played['kind'] == 'saved':
                boards['/'.join(pathlib.Path(played['adapter file']).parts[-3:-1])] = (rewards, costs)
                continue
            name = 'unadapted'
            if played['kind'] == 'steering':
                roll = '' if played.get('look-ahead roll', 'likely') == 'likely' else f', look-ahead on the {played["look-ahead roll"]}'
                name = f'steering x {played["scale"]:g} ({played.get("form", "residual")}{roll})'
            baselines.append({'task': saved['task'], 'name': name, 'rewards': rewards, 'costs': costs,
                              'limit': saved.get('cost limit', 2)})
        for file in sorted(root.rglob('episodes.jsonl')):
            rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
            if rows and 'variant' in rows[0]:
                baselines.append({'task': rows[0]['task'], 'name': f'SafeDreamer {rows[0]["variant"]}',
                                  'rewards': [row['reward'] for row in rows], 'costs': [row['cost'] for row in rows], 'limit': 2})
        for file in sorted(root.rglob('final.json')):
            final = json.loads(file.read_text())
            where = '/'.join(file.parent.parts[-2:])
            searches.append({'task': final['task'], 'arm': arm_of(final['settings']), 'seed': final['settings']['seed'],
                             'chosen': final['chosen'], 'limit': final['settings']['limit'], 'steps': final['simulator steps'],
                             'board': boards.get(where), 'where': f'{root.name}/{where}'})

    rows = []
    for task in sorted({one['task'] for one in searches + baselines}):
        print(f'\n{task}')
        print(f'{"":58s} {"seeds":>5s}  {"as the search chose":^31s}  {"on the scoreboard":^40s}')
        print(f'{"":58s} {"":>5s}  {"reward":>12s} {"cost":>11s} {"within":>6s}  {"reward":>12s} {"cost":>11s} {"within":>6s} {"played":>8s}')
        for arm in sorted({one['arm'] for one in searches if one['task'] == task}):
            mine = sorted((one for one in searches if one['task'] == task and one['arm'] == arm), key=lambda one: one['seed'])
            limit = mine[0]['limit']
            reward, cost = (mean_and_error([one['chosen'][name] for one in mine]) for name in ('reward', 'cost'))
            within = np.mean([one['chosen']['cost'] <= limit for one in mine])
            line = f'{arm[:58]:58s} {len(mine):5d}  {reward[0]:6.2f} ±{reward[1]:5.2f} {cost[0]:5.2f} ±{cost[1]:5.2f} {within:6.0%}'
            row = {'task': task, 'arm': arm, 'seeds': len(mine), 'chosen reward': reward, 'chosen cost': cost, 'chosen within': float(within),
                   'simulator steps': mine[0]['steps'], 'runs': [one['where'] for one in mine]}
            played = [one for one in mine if one['board']]
            if played:
                reward, cost = (mean_and_error([np.mean(one['board'][index]) for one in played]) for index in (0, 1))
                within = np.mean([np.mean(one['board'][1]) <= limit for one in played])
                line += (f'  {reward[0]:6.2f} ±{reward[1]:5.2f} {cost[0]:5.2f} ±{cost[1]:5.2f} {within:6.0%} '
                         f'{len(played):3d} x {len(played[0]["board"][0])}')
                row.update({'board reward': reward, 'board cost': cost, 'board within': float(within),
                            'each seed on the board': [[float(np.mean(one['board'][0])), float(np.mean(one['board'][1]))] for one in played]})
            print(line)
            rows.append(row)
        for one in sorted((one for one in baselines if one['task'] == task), key=lambda one: one['name']):
            reward, cost = mean_and_error(one['rewards']), mean_and_error(one['costs'])
            print(f'{one["name"][:58]:58s} {"":5s}  {"":31s}  {reward[0]:6.2f} ±{reward[1]:5.2f} {cost[0]:5.2f} ±{cost[1]:5.2f} '
                  f'{np.mean(np.array(one["costs"]) <= one["limit"]):6.0%} {len(one["costs"]):8d}')
            rows.append({'task': task, 'baseline': one['name'], 'episodes': len(one['costs']), 'reward': reward, 'cost': cost})
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
