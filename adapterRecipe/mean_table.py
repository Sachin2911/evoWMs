"""The mean table: one adapter whose weights are the mean of the adapters several searches of one task ended with.

Searches of the same task with different seeds end with different weights. Part of the
difference is what the task asks for, which they share, and part is the noise of each
search, which they do not. The mean keeps the first and thins the second. No search and no
episode is needed to make it.

It takes a run folder of finished searches, one sub-folder each, as a sweep of search.py
leaves it. The searches of the task are the sub-folders whose final.json names it. Their
final.npz files must hold adapters of the same form, settings and number of weights; if
they do not, or if fewer than two are found, it stops.

A search fits the cost predictor again after every check (unless --keep-predictor), so the
adapters of one task can come with slightly different predictors. An adapter file holds
one, so the mean adapter is given the mean of theirs, array by array, or with
--predictor first that of the first search. The sizes of the look-ahead numbers are
treated the same way. How far apart they were is printed.

The file it writes is one that adapter.save could have written, so it is played with

    scoreboard.py --kind saved --adapter=FILE

It needs only numpy. Through on_task.sh, which puts the task's name where TASK stands and
hands over a --checkpoint that is not used here:

    bash on_task.sh mean_table.py --runs RUN_FOLDER --task-name TASK --out FILE --task=PointGoal2
"""

import argparse
import json
import pathlib

import numpy as np


def names_task(named, task):
    """Whether `named`, the task as a final.json has it, is the task called `task`, as in PointGoal2."""
    return named == task or f'Safety{task}-v0' in named


def searches_of(runs, task):
    """The final.npz of every search of `task` under the run folder `runs`, in name order."""
    found = []
    for final in sorted(pathlib.Path(runs).glob('*/final.json')):
        if names_task(json.loads(final.read_text())['task'], task):
            found.append(final.with_suffix('.npz'))
    return found


def is_setting(name):
    """Whether an entry of an adapter file is a setting, which every search must share, and not numbers that are averaged."""
    return name not in ('weights', 'sizes') and not name.startswith('predictor_')


def spread(arrays):
    """How far apart the same array is in several files: the largest difference from their mean, as a share of the largest value."""
    stacked = np.stack([np.asarray(array, np.float64) for array in arrays])
    largest = np.abs(stacked).max()
    return float(np.abs(stacked - stacked.mean(0)).max() / largest) if largest else 0.0


def mean_of(files, predictor='mean'):
    """The entries of the mean adapter of the adapter files `files`.

    The weights are the mean. The settings are those of every file, and it stops if one
    file differs. The cost predictor and the sizes are the mean too, or with `predictor`
    as 'first' those of the first file. Also returns how far apart each averaged array was.
    """
    if len(files) < 2:
        raise ValueError(f'a mean needs at least 2 adapters, and {len(files)} were found')
    saved = [dict(np.load(file)) for file in files]
    first = saved[0]
    apart = {}
    mean = {}
    for file, other in zip(files, saved):
        if sorted(other) != sorted(first):
            raise ValueError(f'{file} holds {sorted(other)}, and {files[0]} holds {sorted(first)}')
    for name, value in first.items():
        for file, other in zip(files, saved):
            if other[name].shape != value.shape or other[name].dtype != value.dtype:
                raise ValueError(f'`{name}` is {other[name].dtype} of shape {other[name].shape} in {file}, '
                                 f'and {value.dtype} of shape {value.shape} in {files[0]}')
            if is_setting(name) and not np.array_equal(other[name], value):
                raise ValueError(f'the setting `{name}` is {other[name]} in {file} and {value} in {files[0]}')
        if is_setting(name):
            mean[name] = value
            continue
        apart[name] = spread([other[name] for other in saved])
        if name == 'weights' or predictor == 'mean':
            # Summed with more digits than the file keeps, then written as the file keeps them.
            mean[name] = np.mean([other[name].astype(np.float64) for other in saved], 0).astype(value.dtype)
        else:
            mean[name] = value
    return mean, apart


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', required=True, help='a run folder with one sub-folder per finished search')
    parser.add_argument('--task-name', required=True, help='the task whose searches are averaged, as in PointGoal2')
    parser.add_argument('--out', required=True, help='the adapter file to write')
    parser.add_argument('--predictor', default='mean', choices=('mean', 'first'),
                        help='the cost predictor and sizes the mean adapter is given: the mean of the searches\', or the first search\'s')
    parser.add_argument('--checkpoint', help='not used; on_task.sh hands one to every script')
    args = parser.parse_args()

    files = searches_of(args.runs, args.task_name)
    if len(files) < 2:
        raise SystemExit(f'mean_table.py: {len(files)} searches of {args.task_name} with a final.json in {args.runs}; a mean needs at least 2')
    mean, apart = mean_of(files, args.predictor)
    # Which adapters it is the mean of is kept in the file. adapter.load does not read it.
    np.savez(args.out, **mean, mean_of=np.array([str(file) for file in files]))

    print(f'{args.task_name}: the mean of {len(files)} adapters of form {mean["form"]} with {mean["weights"].size} weights each')
    for file in files:
        print(f'  {file}')
    print('largest difference from the mean, as a share of the largest value: '
          + ', '.join(f'{name} {value:.2g}' for name, value in sorted(apart.items())))
    print(f'the cost predictor and the sizes are {"the mean of theirs" if args.predictor == "mean" else "those of the first"}')
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
