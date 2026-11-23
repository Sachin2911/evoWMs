"""Sets imagined rolls beside what really followed, from the states of an adapter's own episodes.

A search scores an adapter by imagined rolls. If the world model imagines a reward where
the robot really stood still, no score can see a stall, and nothing built on the score can
either (implementation/adapterArchitecture.md, order of work, step 2).

The adapter plays real episodes. Rolls of the world model are then started from states of
those episodes, with the same adapter and the memory it had at each, and every roll is
compared with the decisions that really followed its starting state.

The starting states are put in four groups by the reward that really followed, the lowest
first. A stall is in the lowest group if there was one: a robot that stands in front of a
hazard earns nothing. What is asked is whether the world model imagines little there too.

    real, imagined   reward and cost added up over the decisions of a roll, as a mean
                     over the group. The imagined ones are a mean over several rolls
                     with different random numbers
    unadapted        what the world model imagines the unadapted agent earning from the
                     same states, which is what a candidate is measured against
    speed            how fast the robot really moved (Safety-Gymnasium only)
    rank agreement   between imagined and real over all the starting states
    shares           what a search goes by: the adapter's imagined reward and cost as
                     shares of the unadapted agent's from the same states, beside its
                     real reward and cost as shares of the unadapted agent's own episodes

The adapter is steering written in the form of the adapter with memory, or with --adapter
one that a search saved.

From this folder:

  ../msCli/ms submit -t 2h -n check-stall -- bash on_task.sh check_stall.py --predictor PREDICTOR --task=PointGoal1
"""

import argparse
import json
import os
import pathlib

import numpy as np

import adapter as adapters
from cost_predictor import files_of, read
from imagine import Imagination
from look_ahead import measured_by_stretch
from search import player_of, together


def rank_agreement(one, other):
    """Spearman's rank correlation: 1 if the two put the starting states in the same order, 0 if in unrelated orders."""
    from scipy import stats
    if np.ptp(one) == 0 or np.ptp(other) == 0:
        return None
    return float(stats.spearmanr(one, other).correlation)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True, help='the cost predictor steering works with, and whose episodes are the unadapted agent\'s')
    parser.add_argument('--adapter', help='an adapter with memory saved by a search (default: steering)')
    parser.add_argument('--scale', type=float, default=1.0, help='steering: what its weights are multiplied by')
    parser.add_argument('--episodes', type=int, default=20, help='real episodes the adapter plays')
    parser.add_argument('--first-seed', type=int, default=4000)
    parser.add_argument('--starts', type=int, default=512, help='starting states drawn from those episodes')
    parser.add_argument('--roll', type=int, default=40, help='decisions of a roll')
    parser.add_argument('--draws', type=int, default=4, help='imagined rolls from each starting state')
    args = parser.parse_args()

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    predictor, trained, _ = read(args.predictor, args.checkpoint)
    player = player_of(args.checkpoint)
    frozen = player.frozen
    if args.adapter:
        adapter, weights, fixed = adapters.load(args.adapter, player.action_numbers)
        if adapter.form != 'memory':
            raise SystemExit('this check is for the adapter with memory')
        described = {'adapter file': args.adapter, **adapter.describe()}
    else:
        for_gpu = {name: predictor[name] for name in predictor if name != 'ahead'}
        adapter = adapters.MemoryAdapter(player.action_numbers)
        fixed = {'predictor': for_gpu,
                 'sizes': measured_by_stretch(frozen, for_gpu, files_of(trained, 'training'), adapter.lengths, adapter.stop)}
        weights = adapter.steering(args.scale)
        described = {'steering': args.scale, **adapter.describe()}
    weights = np.asarray(weights, np.float32)
    act = adapter.compiled(frozen)

    # The real episodes of the adapter.
    if hasattr(player, 'watch'):
        player.watch = True
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    played = list(player.play(seeds, memory=adapter.start(),
                              adapter=lambda state, proposed, held: act(state, proposed, held, fixed, weights)))
    decisions = sum(len(episode['cost']) - 1 for episode in played)
    adapted = {name: sum(float(episode[name].sum()) for episode in played) / decisions for name in ('reward', 'cost')}

    # The unadapted agent's own episodes: those the predictor began with.
    alone = [np.load(file) for part in ('validation', 'training') for file in files_of(trained, part)]
    alone_decisions = sum(len(episode['cost']) - 1 for episode in alone)
    usual = {name: sum(float(episode[name].sum()) for episode in alone) / alone_decisions for name in ('reward', 'cost')}

    # Starting states with room for a whole roll after them, and what really followed each.
    random = np.random.default_rng(0)
    room = [(number, entry) for number, episode in enumerate(played) for entry in range(len(episode['cost']) - args.roll)]
    chosen = sorted(room[pick] for pick in random.choice(len(room), min(args.starts, len(room)), replace=False))
    parts, memories, real = [], [], {'reward': [], 'cost': [], 'speed': []}
    for number, episode in enumerate(played):
        entries = np.array([entry for of_episode, entry in chosen if of_episode == number], int)
        if not len(entries):
            continue
        parts.append(frozen.kept(episode, entries))
        memories.append(episode['memory'][entries])
        # Entry t + 1 of an episode holds what came of the action of entry t.
        after = entries[:, None] + 1 + np.arange(args.roll)
        real['reward'].append(episode['reward'][after].sum(1))
        real['cost'].append(episode['cost'][after].sum(1))
        if 'speed' in episode:
            real['speed'].append(episode['speed'][after].mean(1))
    real = {name: np.concatenate(rows) for name, rows in real.items() if rows}
    state, memory = frozen.written_out(together(parts)), np.concatenate(memories)

    # The same decisions, imagined: with the adapter, and with none.
    imagination = Imagination(frozen, adapter, fixed, state)

    def imagined_with(weights):
        rolls = [imagination.roll(weights, args.roll, seed, memory=memory) for seed in range(args.draws)]
        return {name: np.mean([roll[name].sum(0) for roll in rolls], 0) for name in ('reward', 'cost')}

    imagined, unadapted = imagined_with(weights), imagined_with(np.zeros_like(weights))

    order = np.argsort(real['reward'], kind='stable')
    groups = []
    print(f'{player.task}: {len(played)} real episodes, reward {np.mean([episode["reward"].sum() for episode in played]):.2f}, '
          f'cost {np.mean([episode["cost"].sum() for episode in played]):.2f}; {len(order)} starting states, '
          f'rolls of {args.roll} decisions, {args.draws} imagined rolls from each', flush=True)
    print('group by real reward     real reward  imagined  unadapted imagined   real cost  imagined   speed')
    for number, members in enumerate(np.array_split(order, 4)):
        row = {'real reward': float(real['reward'][members].mean()), 'imagined reward': float(imagined['reward'][members].mean()),
               'unadapted imagined reward': float(unadapted['reward'][members].mean()),
               'real cost': float(real['cost'][members].mean()), 'imagined cost': float(imagined['cost'][members].mean()),
               'unadapted imagined cost': float(unadapted['cost'][members].mean()),
               'speed': float(real['speed'][members].mean()) if 'speed' in real else None}
        groups.append(row)
        speed = '' if row['speed'] is None else f'{row["speed"]:8.3f}'
        print(f'  quarter {number + 1} {"(lowest)" if number == 0 else "        "}  {row["real reward"]:11.3f} {row["imagined reward"]:9.3f} '
              f'{row["unadapted imagined reward"]:19.3f} {row["real cost"]:11.3f} {row["imagined cost"]:9.3f} {speed}', flush=True)

    result = {
        'task': player.task, 'adapter': described, 'episodes': len(played), 'starting states': len(order),
        'roll': args.roll, 'draws': args.draws,
        'real episodes': {'reward': [float(episode['reward'].sum()) for episode in played],
                          'cost': [float(episode['cost'].sum()) for episode in played]},
        'groups by real reward': groups,
        'rank agreement': {'reward': rank_agreement(imagined['reward'], real['reward']),
                           'cost': rank_agreement(imagined['cost'], real['cost'])},
        'shares': {
            # As search.py works a share out: 1 is as much as the unadapted agent.
            'imagined reward': float(1 + (imagined['reward'].mean() - unadapted['reward'].mean()) / max(abs(unadapted['reward'].mean()), 1e-9)),
            'imagined cost': float(imagined['cost'].mean() / max(unadapted['cost'].mean(), 1e-9)),
            'real reward': adapted['reward'] / usual['reward'],
            'real cost': adapted['cost'] / usual['cost'],
        },
        'a decision': {'real reward, adapter': adapted['reward'], 'real reward, unadapted': usual['reward'],
                       'real cost, adapter': adapted['cost'], 'real cost, unadapted': usual['cost']},
    }
    agreement, shares = result['rank agreement'], result['shares']
    print('rank agreement of imagined with real over the starting states: '
          + ', '.join(f'{name} {"none" if value is None else f"{value:+.2f}"}' for name, value in agreement.items()))
    print(f'shares of the unadapted agent: reward {shares["imagined reward"]:.2f} imagined and {shares["real reward"]:.2f} real, '
          f'cost {shares["imagined cost"]:.2f} imagined and {shares["real cost"]:.2f} real')
    (out / 'check_stall.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
