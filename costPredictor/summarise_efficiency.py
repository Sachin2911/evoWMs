"""Prints how the cost predictors' scores change with the number of episodes they learn from.

  python3 summarise_efficiency.py FOLDER

FOLDER holds <checkpoint>/episodes_<N>/results.json, as written by
train_cost_predictor.py --labelled-episodes N. Scores are AUROC on the same 50 test
episodes, for decisions where the agent was not already being charged (0.5 is chance).
"""

import json
import pathlib
import re
import sys

PREDICTORS = ['deter', 'latent', 'privileged', 'shuffled']


def main():
    folder = pathlib.Path(sys.argv[1])
    for model in ('network', 'linear'):
        print(f'\nAUROC when not already being charged, {model} predictor: H = 5 / 15 / 25')
        print(f"{'task':13s} {'episodes':>8s} {'steps':>8s}  " + '  '.join(f'{name:>21s}' for name in PREDICTORS))
        paths = sorted(folder.glob('*/episodes_*/results.json'), key=lambda path: (path.parts[-3], int(path.parts[-2].split('_')[1])))
        for path in paths:
            results = json.loads(path.read_text())
            task = re.search(r'Safety(\w+)-v0', results['task']).group(1)
            labelled = results['episodes_in']['train'] + results['episodes_in']['validation']
            cells = [' / '.join(f"{results['horizons'][h]['predictors'][name][model]['auroc_when_calm']:.3f}" for h in ('5', '15', '25'))
                     for name in PREDICTORS if name in results['horizons']['5']['predictors']]
            print(f"{task:13s} {labelled:8d} {results['simulator_steps_labelled']:8d}  " + '  '.join(f'{cell:>21s}' for cell in cells))


if __name__ == '__main__':
    main()
