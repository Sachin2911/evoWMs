"""Prints the standing table: everything measured on the development seeds, side by side.

It reads run folders that were fetched to the laptop and plays nothing. In them it looks
for what scoreboard.py wrote (scoreboard.json) and for what safeDreamerBaselines/evaluate.py
wrote for SafeDreamer's own checkpoints (episodes.jsonl). An adapter that a search saved is
shown with that search's settings and with the real steps the search used, which are read
from the final.json beside the adapter if that run folder was fetched too.

    python3 standings.py ../results/evoWMs-round0-* ../results/evoWMs-round1-*

Columns: the mean reward and cost of an episode with their standard errors, the share of
episodes within the cost limit, the highest cost of one episode, the share of decisions at
which an action number was moved by more than 0.1, and the real simulator steps that were
spent to get the adapter (those behind its cost predictor included).
"""

import argparse
import json
import pathlib

import numpy as np

COST_LIMIT = 2
RUNS_ON_CLUSTER = '/home-mscluster/smohan/runs/'


def mean_and_error(values):
    values = np.asarray(values, np.float64)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def search_of(adapter_file, fetched):
    """The final.json of the search that saved `adapter_file`, if its run folder is among the fetched ones."""
    if not adapter_file.startswith(RUNS_ON_CLUSTER):
        return None
    run, _, rest = adapter_file[len(RUNS_ON_CLUSTER):].partition('/')
    for folder in fetched:
        if folder.name == run:
            final = (folder / rest).with_name('final.json')
            return json.loads(final.read_text()) if final.exists() else None
    return None


def from_scoreboard(file, fetched):
    saved = json.loads(file.read_text())
    played, summary = saved['played'], saved['summary']
    steps = ''
    if played['kind'] == 'none':
        name = 'unadapted'
    elif played['kind'] == 'steering':
        name = f'steering x {played["scale"]:g}, predictor of {played["episodes behind the predictor"]} episodes'
        steps = played['episodes behind the predictor'] * 1000
    elif played['kind'] == 're-rank':
        name = (f're-rank, reward {played["reward weight"]:g}, cost {played["cost weight"]:g}, '
                f'predictor of {played["episodes behind the predictor"]} episodes')
        steps = played['episodes behind the predictor'] * 1000
    else:
        search = search_of(played['adapter file'], fetched)
        name = 'saved: ' + '/'.join(pathlib.Path(played['adapter file']).parts[-3:-1])
        if search:
            settings = search['settings']
            changed = [f'{key} {value}' for key, value in sorted(settings.items())
                       if key in ('form', 'size', 'generations', 'keep_predictor', 'candidates', 'starts', 'roll', 'failures')
                       and value != DEFAULTS[key]]
            name = f'search, {settings["search"]}' + ''.join(f', {one}' for one in changed) + f', seed {settings["seed"]}'
            steps = search['simulator steps']
    rewards = [row['reward'] for row in saved['each episode']]
    costs = [row['cost'] for row in saved['each episode']]
    return {'task': saved['task'], 'name': name, 'rewards': rewards, 'costs': costs, 'steps': steps,
            'moved': summary['decisions at which a number was moved by more than 0.1']}


def from_evaluate(file):
    rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
    first = rows[0]
    return {'task': first['task'], 'name': f'SafeDreamer {first["variant"]}, {first["checkpoint"].rsplit("_", 1)[-1][:-5]}',
            'rewards': [row['reward'] for row in rows], 'costs': [row['cost'] for row in rows], 'steps': 'trained on cost',
            'moved': None}


# The settings of search.py that make recipe version 0; a search is named by where it differs.
DEFAULTS = {'form': 'residual', 'size': 'tens', 'generations': 5, 'keep_predictor': False, 'candidates': 16,
            'starts': 256, 'roll': 40, 'failures': 0.25}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folders', nargs='+', type=pathlib.Path, help='fetched run folders')
    parser.add_argument('--json', type=pathlib.Path, help='also write the rows to this file')
    args = parser.parse_args()

    rows = []
    for folder in args.folders:
        rows += [from_scoreboard(file, args.folders) for file in sorted(folder.rglob('scoreboard.json'))]
        rows += [from_evaluate(file) for file in sorted(folder.rglob('episodes.jsonl'))]
    for row in rows:
        costs = np.array(row['costs'])
        row.update({'episodes': len(costs), 'reward': mean_and_error(row['rewards']), 'cost': mean_and_error(costs),
                    'within': float((costs <= COST_LIMIT).mean()), 'highest': float(costs.max())})
    for task in sorted({row['task'] for row in rows}):
        print(f'\n{task}')
        print(f'{"":64s} {"episodes":>8s} {"reward":>14s} {"cost":>14s} {"within":>7s} {"highest":>8s} {"moved":>6s} {"steps":>16s}')
        for row in sorted((row for row in rows if row['task'] == task), key=lambda row: row['name']):
            moved = '' if row['moved'] is None else f'{row["moved"]:.0%}'
            steps = f'{row["steps"]:,}' if isinstance(row['steps'], int) else row['steps']
            print(f'{row["name"]:64s} {row["episodes"]:8d} {row["reward"][0]:7.2f} ± {row["reward"][1]:4.2f} '
                  f'{row["cost"][0]:7.2f} ± {row["cost"][1]:4.2f} {row["within"]:7.0%} {row["highest"]:8.0f} {moved:>6s} {steps:>16s}')
    if args.json:
        args.json.write_text(json.dumps([{name: value for name, value in row.items() if name not in ('rewards', 'costs')}
                                         for row in rows], indent=1) + '\n')


if __name__ == '__main__':
    main()
