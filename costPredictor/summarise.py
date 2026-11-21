"""Prints the cost-predictor results of every task side by side.

  python3 summarise.py FOLDER

FOLDER holds one sub-folder per checkpoint, each with the results.json that
train_cost_predictor.py wrote. Every number is an AUROC on the test episodes: the chance
that a decision followed by a cost gets a higher score than one that is not (0.5 is chance).
"""

import json
import pathlib
import re
import sys

PREDICTORS = ['latent', 'deter', 'stoch', 'privileged', 'shuffled']


def main():
    folder = pathlib.Path(sys.argv[1])
    for when, key in (('all decisions', 'auroc'), ('decisions not already being charged', 'auroc_when_calm')):
        print(f'\nAUROC, {when}: linear / network')
        print(f"{'task':13s} {'H':>2s} {'cost ahead':>10s}  " + '  '.join(f'{name:>13s}' for name in PREDICTORS))
        for path in sorted(folder.glob('*/results.json')):
            results = json.loads(path.read_text())
            task = re.search(r'Safety(\w+)-v0', results['task']).group(1)
            for horizon, entry in results['horizons'].items():
                share = entry['share_followed_by_cost' if key == 'auroc' else 'share_followed_by_cost_when_calm']
                cells = [f"{entry['predictors'][name]['linear'][key]:.3f} / {entry['predictors'][name]['network'][key]:.3f}"
                         for name in PREDICTORS]
                print(f'{task:13s} {horizon:>2s} {100 * share:9.1f}%  ' + '  '.join(cells))


if __name__ == '__main__':
    main()
