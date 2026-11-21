"""Part A: compares CPU-rendered and GPU-rendered evaluations of the same checkpoints.

Reads results/<renderer>-<repeat>/<checkpoint>.jsonl, as written on the rented machines.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

root = sys.argv[1]
# Averages from the cluster's GPU-rendered recording run (job 92099), for context only.
CLUSTER = {
    'PointGoal1 dreamerv3 s0': (26.90, 54.40), 'PointGoal1 bsrp_lag s0': (21.74, 0.88),
    'PointGoal1 bsrp_lag s5': (21.36, 0.78), 'PointGoal1 bsrp_lag s10': (21.62, 1.80),
    'CarGoal1 dreamerv3 s0': (34.67, 54.32), 'CarGoal1 bsrp_lag s0': (26.89, 0.14),
    'PointButton1 dreamerv3 s0': (27.16, 96.40), 'PointButton1 bsrp_lag s0': (5.82, 0.62),
    'PointPush1 dreamerv3 s0': (23.75, 39.54), 'PointPush1 bsrp_lag s0': (16.05, 1.24),
    'PointGoal2 dreamerv3 s0': (23.19, 189.80), 'PointGoal2 bsrp_lag s0': (13.79, 0.48),
}
ORDER = list(CLUSTER)


def short(name):
    m = re.search(r'_(dreamerv3|bsrp_lag)_safetygym_Safety(\w+)-v0_(\d+)', name)
    return f'{m.group(2)} {m.group(1)} s{m.group(3)}'


runs = {}   # (renderer, repeat, checkpoint) -> {seed: (reward, cost, seconds)}
for path in glob.glob(os.path.join(root, '*', '*.jsonl')):
    renderer, repeat = os.path.basename(os.path.dirname(path)).rsplit('-', 1)
    rows = [json.loads(line) for line in open(path)]
    runs[renderer, int(repeat), short(os.path.basename(path))] = {r['seed']: (r['reward'], r['cost'], r['seconds']) for r in rows}

print(f'{len(runs)} result files; episodes per file: {sorted({len(v) for v in runs.values()})}')
z_all = {'reward': [], 'cost': []}
replay = {'egl': [0, 0], 'osmesa': [0, 0]}
print()
print('checkpoint                 | replayed (of 50)  | s/episode   | reward: GPU   CPU   cluster | cost: GPU    CPU   cluster | CPU-GPU reward (z) | CPU-GPU cost (z)')
print('                           | GPU     CPU       | GPU   CPU   |')
for ck in ORDER:
    have = all((r, k, ck) in runs and len(runs[r, k, ck]) == 50 for r in ('egl', 'osmesa') for k in (1, 2))
    if not have:
        print(f'{ck:26s} | INCOMPLETE: ' + ', '.join(f'{r}-{k}: {len(runs.get((r, k, ck), {}))}' for r in ('egl', 'osmesa') for k in (1, 2)))
        continue
    same = {}
    for r in ('egl', 'osmesa'):
        same[r] = sum(runs[r, 1, ck][s][:2] == runs[r, 2, ck][s][:2] for s in range(50))
        replay[r][0] += same[r]; replay[r][1] += 50
    # Per seed, the average of the two repeats; then CPU minus GPU, paired by seed.
    per = {r: np.array([[(runs[r, 1, ck][s][i] + runs[r, 2, ck][s][i]) / 2 for i in (0, 1)] for s in range(50)]) for r in ('egl', 'osmesa')}
    secs = {r: np.median([runs[r, k, ck][s][2] for k in (1, 2) for s in range(50)]) for r in ('egl', 'osmesa')}
    out = []
    for i, name in enumerate(('reward', 'cost')):
        d = per['osmesa'][:, i] - per['egl'][:, i]
        se = d.std(ddof=1) / math.sqrt(len(d))
        z = d.mean() / se if se > 0 else 0.0
        z_all[name].append(z)
        out.append(f'{d.mean():+6.2f} ({z:+.1f})')
    g, c = per['egl'].mean(0), per['osmesa'].mean(0)
    print(f'{ck:26s} | {same["egl"]:3d}     {same["osmesa"]:3d}       | {secs["egl"]:4.1f}  {secs["osmesa"]:5.1f} '
          f'| {g[0]:11.2f} {c[0]:6.2f} {CLUSTER[ck][0]:7.2f} | {g[1]:9.2f} {c[1]:7.2f} {CLUSTER[ck][1]:7.2f} | {out[0]:>14s}     | {out[1]:>14s}')
print()
for r, (a, b) in replay.items():
    if b: print(f'{r}: {a} of {b} episodes identical between the two repeats ({100 * a / b:.0f}%)')
for name, zs in z_all.items():
    zs = np.array(zs)
    if len(zs):
        print(f'{name}: z-scores {np.round(zs, 1).tolist()} | combined z {zs.sum() / math.sqrt(len(zs)):+.2f} | |z| > 2 in {(abs(zs) > 2).sum()} of {len(zs)}')
