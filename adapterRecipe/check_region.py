"""Asks whether an adapter still keeps the gripper out of the region when the region is moved or resized.

A Meta-World adapter never sees the region. It reads what the cost predictor says of each
imagined future, so where the region is and how large it is are held by the predictor
alone. This script moves or resizes the region, fits the predictor of the new one, leaves
the adapter's weights as the search left them, and plays.

Nothing is played to fit the new predictor. The agent never sees the cost, so the saved
episodes of the unadapted agent are what they would have been with any region, and the
cost of another region is worked out from where the gripper was (frozen_tdmpc2.regions
does the same for other radii).

The region stays a ball. Its centre is put

    --along A    this share of the way from where the gripper starts to the first thing it
                 has to reach. 0.5, the middle, is where the adapters were searched
    --up U       and this many metres higher

and --radius R is its radius in metres (default: the task's own).

What it does, in order:

    1  writes the saved episodes of --episodes out again with the cost of the new region
    2  has cost_predictor.py fit a predictor to them, split as prepare_task.sh splits them
    3  measures the sizes of the look-ahead numbers with that predictor, as search.py does
    4  plays --fresh episodes from --first-seed three times: the unadapted agent, the
       adapter of --adapter exactly as it was saved, and the same weights with the new
       predictor and sizes
    5  counts the cost of all three for the new region

The second of the three says what the adapter does when nothing tells it the region has
changed. The third is saved as adapter.npz, in the form adapter.save writes, so it can be
played elsewhere.

    ../msCli/ms submit --grid "--task=mw-button-press-topdown --radius=0.10,0.13,0.16" -t 2h -n region -- \\
        bash on_task.sh check_region.py --episodes /datasets/smohan/adapterRecipe/episodes/TASK \\
        --adapter ../demo/adapters/TASK.npz
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

import adapter as adapters
from cost_predictor import files_of, read
from frozen_tdmpc2 import Player
from look_ahead import measured_by_stretch


def centre_of(start, target, along, up):
    """The centre of the region: `along` of the way from `start` to `target`, and `up` metres higher."""
    return (start + along * (target - start) + np.array([0.0, 0.0, up])).astype(np.float32)


def distance_from(episode, along, up):
    """How far the gripper is from the centre of the region at every entry of an episode."""
    return np.linalg.norm(episode['gripper'] - centre_of(episode['start'], episode['target'], along, up), axis=1)


def rewritten(folder, out, along, up, radius):
    """Writes the episodes of `folder` into `out` with the cost of the new region, and nothing else changed."""
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(folder.glob('seed_*.npz'))
    for file in files:
        episode = dict(np.load(file))
        distance = distance_from(episode, along, up)
        if distance[0] < radius:
            raise SystemExit(f'{file.name}: the gripper starts inside this region, {distance[0]:.3f} m from its centre')
        episode.update(distance=distance, cost=(distance < radius).astype(np.float32), radius=radius,
                       centre=centre_of(episode['start'], episode['target'], along, up))
        np.savez_compressed(out / file.name, **episode)
    collected = json.loads((folder / 'run.json').read_text())
    collected['region'] = {'along': along, 'up': up, 'radius': radius, 'rewritten from': str(folder)}
    (out / 'run.json').write_text(json.dumps(collected) + '\n')
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', required=True, help='the saved episodes of the unadapted agent (frozen_tdmpc2.py)')
    parser.add_argument('--adapter', required=True, help='an adapter with memory, saved by adapter.save')
    parser.add_argument('--along', type=float, default=0.5)
    parser.add_argument('--up', type=float, default=0.0)
    parser.add_argument('--radius', type=float)
    parser.add_argument('--fresh', type=int, default=100, help='episodes each of the three plays')
    parser.add_argument('--first-seed', type=int, default=5000)
    args = parser.parse_args()

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    player = Player(args.checkpoint)
    frozen = player.frozen
    if args.radius is None:
        args.radius = player.radius
    region = {'along': args.along, 'up': args.up, 'radius': args.radius}

    # 1 and 2: the saved episodes with the cost of this region, and the predictor fitted to them.
    count = rewritten(pathlib.Path(args.episodes), out / 'episodes', args.along, args.up, args.radius)
    subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('cost_predictor.py')), '--episodes', str(out / 'episodes'),
                    '--train', '20', '--out', str(out / 'predictor')], check=True)
    predictor, trained, _ = read(out / 'predictor' / 'cost_predictor.npz', args.checkpoint)
    predictor = {name: predictor[name] for name in predictor if name != 'ahead'}

    # 3: the adapter's weights stay; the predictor and the sizes measured with it are the new region's.
    state_numbers = int(predictor['weights'].shape[0])
    adapter, weights, as_saved = adapters.load(args.adapter, player.action_numbers, state_numbers)
    if adapter.form != 'memory':
        raise SystemExit('check_region.py is for the adapter with memory')
    weights = np.asarray(weights, np.float32)
    moved_to = {'predictor': predictor,
                'sizes': measured_by_stretch(frozen, predictor, files_of(trained, 'training'), adapter.lengths, adapter.stop,
                                             adapter.roll)}
    adapters.save(out / 'adapter.npz', adapter, weights, moved_to)

    # 4 and 5: the three plays, on the same seeds, with the cost counted for this region.
    act = adapter.compiled(frozen)
    seeds = list(range(args.first_seed, args.first_seed + args.fresh))
    arms = {'unadapted': None, 'adapter as saved': as_saved, 'adapter, new predictor': moved_to}
    played = {}
    for name, fixed in arms.items():
        with_adapter = {} if fixed is None else {
            'adapter': lambda state, proposed, held, fixed=fixed: act(state, proposed, held, fixed, weights),
            'memory': adapter.start()}
        rows = []
        for episode in player.play(seeds, **with_adapter):
            far = np.abs(np.clip(episode['action'], -1, 1) - np.clip(episode['proposed'], -1, 1)).max(1) > 0.1
            rows.append({'seed': int(episode['seed']), 'reward': float(episode['reward'].sum()),
                         'cost': float((distance_from(episode, args.along, args.up) < args.radius).sum()),
                         'cost of the region it was searched with': float(episode['cost'].sum()),
                         'succeeded': bool(episode['success'].max()), 'moved by more than 0.1': float(far[:-1].mean())})
        played[name] = rows
        print(f'{name}: {len(rows)} episodes played', flush=True)

    usual = float(np.mean([row['reward'] for row in played['unadapted']]))
    summary = {}
    print(f'\n{player.task}: region {args.along:g} of the way, {args.up:+g} m up, radius {args.radius:g} m '
          f'(searched with 0.5, +0, {player.radius:g}). Predictor fitted to {count} saved episodes: '
          f'AUROC {trained["test"]["AUROC"]:.3f}, costly rows {trained["costly share of training rows"]:.1%}')
    print(f'{"":24s} {"reward":>7s} {"kept":>5s} {"cost":>6s} {"episodes with no cost":>22s} {"succeeded":>10s} {"moved":>6s}')
    for name, rows in played.items():
        mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'seed'}
        summary[name] = {**mean, 'reward kept': mean['reward'] / usual, 'within the limit': mean['cost'] <= player.cost_limit,
                         'episodes with no cost': float(np.mean([row['cost'] == 0 for row in rows]))}
        print(f'{name:24s} {mean["reward"]:7.0f} {mean["reward"] / usual:5.0%} {mean["cost"]:6.2f} '
              f'{summary[name]["episodes with no cost"]:22.0%} {mean["succeeded"]:10.0%} {mean["moved by more than 0.1"]:6.0%}')

    (out / 'check_region.json').write_text(json.dumps({
        'task': player.task, 'checkpoint': pathlib.Path(args.checkpoint).name, 'adapter': args.adapter, 'region': region,
        'searched with': {'along': 0.5, 'up': 0.0, 'radius': player.radius}, 'cost limit': player.cost_limit,
        'predictor': {'AUROC': trained['test']['AUROC'], 'costly share of training rows': trained['costly share of training rows']},
        'summary': summary, 'each episode': played}, indent=1) + '\n')


if __name__ == '__main__':
    main()
