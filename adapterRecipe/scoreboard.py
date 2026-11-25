"""The scoreboard: plays one agent for many episodes on the development seeds and writes down how it did.

Every adapter of every round is measured here, in the same way and on the same seeds, so
that the numbers of different rounds can be set side by side. The seeds are 5000 upward.
They are development seeds: the final evaluation seeds are not played until the recipe is
frozen. SafeDreamer's own checkpoints are played on the same seeds with
safeDreamerBaselines/evaluate.py --first-seed 5000.

What is played is chosen with --kind:

    none        the frozen agent alone
    steering    the hand-written adapter (adapter.steering), with its weights times --scale.
                With --form memory it is written in the form of the adapter with memory
                (MemoryAdapter.steering), which reads the look-ahead by stretches
    re-rank     the re-rank over pushed actions, with --reward-weight and --cost-weight
    saved       an adapter saved by adapter.save, given with --adapter. With --strength S
                its gains and shifts are multiplied by S before they are applied to the
                action; its weights and its memory are left as they are

All but `saved` take the cost predictor from --predictor and measure the sizes of the
look-ahead numbers on that predictor's training episodes. A `saved` adapter brings its own.

Written to the run folder:

    scoreboard.json    reward and cost of every episode, and over all of them: the means
                       with their standard errors, the share of episodes within the cost
                       limit, the cost by the kind of object that charged it, the robot's
                       speed, and how often the adapter moved the action
    scoreboard.npz     one row per episode and one column per decision of what
                       episodes.py watches (reward, cost by kind, speed, place, distances),
                       for looking closer afterwards without playing again
    episodes/          with --keep N, the first N episodes whole, internal states included

From this folder:

  ../msCli/ms submit -t 2h -n scoreboard -- apptainer exec --nv --bind /datasets/smohan \\
      /home-mscluster/smohan/images/safeDreamer.sif python3 scoreboard.py --kind steering \\
      --predictor RUN_FOLDER/cost_predictor.npz --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np

import adapter as adapters
from cost_predictor import files_of, read
from episodes import make_folder, save, states
from look_ahead import look_ahead, measured_by_stretch, sizes
from search import player_of


def mean_and_error(values):
    """The mean and its standard error."""
    values = np.asarray(values, np.float64)
    return [float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))]


def measured_sizes(frozen, predictor, files, decisions, push):
    """The sizes of the look-ahead numbers over every decision of the episodes in `files`."""
    def look(state, proposed, predictor):
        return look_ahead(frozen, predictor, state, proposed, decisions, push)

    look = frozen.compiled(look)
    numbers = []
    for file in files:
        saved = np.load(file)
        numbers.append(look(states(saved), saved['proposed'], predictor))
    return sizes(np.concatenate(numbers))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--kind', required=True, choices=('none', 'steering', 're-rank', 'saved'))
    parser.add_argument('--predictor', help='a cost_predictor.npz, for every kind but `saved`')
    parser.add_argument('--tail', help='a tail.npz of tail.py, whose cost to come the look-ahead adds to every cost it reads: '
                                       'for steering of form memory, and for a `saved` adapter with memory, where it takes the '
                                       'place of the tail saved with the adapter (default: none, or the saved one)')
    parser.add_argument('--adapter', help='a file written by adapter.save, for kind `saved`')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='saved: what the adapter\'s gains and shifts are multiplied by before they are applied to the action')
    parser.add_argument('--scale', type=float, default=1.0, help='steering: what its weights are multiplied by')
    parser.add_argument('--form', default='residual', choices=('residual', 'memory'), help='steering: the form it is written in')
    parser.add_argument('--look', default='likely', choices=('likely', 'mean'),
                        help='steering of form memory: what its look-ahead rolls on')
    parser.add_argument('--reward-weight', type=float, default=0.0, help='re-rank: the weight of a reward change')
    parser.add_argument('--cost-weight', type=float, default=1.0, help='re-rank: the weight of a cost change')
    parser.add_argument('--decisions', type=int, default=10, help='how many decisions the look-ahead lasts')
    parser.add_argument('--push', type=float, default=1.0, help='how far the look-ahead pushes a number towards its limit')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--first-seed', type=int, default=5000)
    parser.add_argument('--keep', type=int, default=0, help='how many episodes to save whole, internal states included')
    parser.add_argument('--limit', type=float,
                        help='the cost limit: the highest mean episode cost that counts as safe (default: the task\'s own)')
    args = parser.parse_args()
    if args.strength != 1.0 and args.kind != 'saved':
        parser.error('--strength is for --kind saved')

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    # DreamerV3 on Safety-Gymnasium or TD-MPC2 on Meta-World, by the checkpoint's name.
    player = player_of(args.checkpoint)
    if hasattr(player, 'watch'):
        player.watch = True
    frozen = player.frozen
    action_numbers = player.action_numbers
    if args.limit is None:
        args.limit = player.cost_limit
    described = {'kind': args.kind}
    act = None
    with_tail = {}
    if args.tail:
        if not (args.kind == 'saved' or (args.kind == 'steering' and args.form == 'memory')):
            raise SystemExit('--tail is for an adapter with memory: --kind saved, or --kind steering --form memory')
        from tail import read as read_tail
        with_tail = {'tail': read_tail(args.tail, args.checkpoint)[0]}
    if args.kind == 'saved':
        probe = dict(np.load(args.adapter))
        state_numbers = int(probe['predictor_weights'].shape[0])
        adapter, weights, fixed = adapters.load(args.adapter, action_numbers, state_numbers)
        if args.strength != 1.0 and adapter.form != 'memory':
            raise SystemExit(f'scoreboard.py: --strength is for an adapter of form memory, and {args.adapter} is of form {adapter.form}')
        if with_tail and adapter.form != 'memory':
            raise SystemExit('--tail is for an adapter with memory, and the saved one has none')
        # The sizes saved with the adapter were measured with the tail it was searched with.
        described.update({'adapter file': args.adapter, **adapter.describe(), 'strength': args.strength})
        if 'tail' in fixed:
            described['tail'] = 'saved with the adapter'
        if with_tail:
            described['tail'] = args.tail
            described['tail saved with the adapter'] = (
                'none' if 'tail' not in fixed else 'the same' if all(
                    np.array_equal(fixed['tail'].get(name), value) for name, value in with_tail['tail'].items()) else 'another')
            if described['tail saved with the adapter'] != 'the same':
                print(f'the adapter was saved with {"no" if "tail" not in fixed else "another"} tail, so its sizes were not '
                      f'measured with the tail of {args.tail}', flush=True)
            fixed = {**fixed, **with_tail}
    elif args.kind == 'steering' and args.form == 'memory':
        predictor, trained, _ = read(args.predictor, args.checkpoint)
        for_gpu = {name: predictor[name] for name in predictor if name != 'ahead'}
        adapter = adapters.MemoryAdapter(action_numbers, roll=args.look)
        fixed = {'predictor': for_gpu, **with_tail,
                 'sizes': measured_by_stretch(frozen, for_gpu, files_of(trained, 'training'),
                                              adapter.lengths, adapter.stop, adapter.roll, with_tail.get('tail'))}
        weights = adapter.steering(args.scale)
        described.update({
            'scale': args.scale, 'predictor': args.predictor, **({'tail': args.tail} if args.tail else {}), **adapter.describe(),
            'episodes behind the predictor': trained['files']['training'][2] + trained['files']['validation'][2]})
    elif args.kind != 'none':
        predictor, trained, _ = read(args.predictor, args.checkpoint)
        for_gpu = {name: predictor[name] for name in ('weights', 'offset', 'mean_cost')}
        state_numbers = int(predictor['weights'].shape[0])
        training = files_of(trained, 'training')
        fixed = {'predictor': for_gpu, 'sizes': measured_sizes(frozen, for_gpu, training, args.decisions, args.push)}
        if args.kind == 'steering':
            adapter = adapters.Adapter('residual', 'tens', action_numbers, state_numbers, args.decisions, args.push)
            weights = (args.scale * adapters.steering(action_numbers)).astype(np.float32)
            described['scale'] = args.scale
        else:
            adapter = adapters.Adapter('re-rank', None, action_numbers, state_numbers, args.decisions, args.push)
            weights = np.array([args.reward_weight, args.cost_weight], np.float32)
            described.update({'reward weight': args.reward_weight, 'cost weight': args.cost_weight})
        described.update({
            'predictor': args.predictor, **adapter.describe(),
            # The real episodes the predictor was trained and set on are part of what this adapter cost.
            'episodes behind the predictor': trained['files']['training'][2] + trained['files']['validation'][2],
            'sizes': [float(size) for size in fixed['sizes']]})
    memory = None
    if args.kind != 'none':
        # At strength 1 the adapter is made exactly as it was before there was a strength.
        compiled = adapter.compiled(frozen) if args.strength == 1.0 else adapter.compiled(frozen, strength=args.strength)
        weights = np.asarray(weights, np.float32)
        if adapter.form == 'memory':
            memory = adapter.start()

            def act(state, proposed, held):
                return compiled(state, proposed, held, fixed, weights)
        else:
            def act(state, proposed):
                return compiled(state, proposed, fixed, weights)

    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    if args.keep:
        make_folder(out / 'episodes')
    started = time.perf_counter()
    rows, by_decision = [], {}
    for number, episode in enumerate(player.play(seeds, adapter=act, memory=memory)):
        if number < args.keep:
            save(episode, out / 'episodes')
        moved = (np.clip(episode['action'], -1, 1) != np.clip(episode['proposed'], -1, 1)).any(1)
        far = np.abs(np.clip(episode['action'], -1, 1) - np.clip(episode['proposed'], -1, 1)).max(1) > 0.1
        watched = {name: episode[name] for name in episode
                   if name in ('reward', 'cost', 'speed', 'x', 'y', 'goals', 'success', 'distance') or name.startswith(('cost_', 'near_'))}
        watched['moved'] = moved.astype(np.float32)
        for name, values in watched.items():
            by_decision.setdefault(name, []).append(values)
        rows.append({
            'seed': int(episode['seed']), 'reward': float(episode['reward'].sum()), 'cost': float(episode['cost'].sum()),
            'moved': float(moved[:-1].mean()), 'moved by more than 0.1': float(far[:-1].mean()),
            **{name: float(episode[name].sum()) for name in episode if name.startswith('cost_')}})
        # What only one of the two simulators reports.
        if 'goals' in episode:
            rows[-1].update({'goals': float(episode['goals'].sum()), 'speed': float(episode['speed'][1:].mean())})
        if 'success' in episode:
            rows[-1].update({'success': float(episode['success'][-1]), 'succeeded at some decision': float(episode['success'].max())})
        if (number + 1) % 25 == 0:
            costs = [row['cost'] for row in rows]
            print(f'{number + 1} episodes: reward {np.mean([row["reward"] for row in rows]):.2f}, cost {np.mean(costs):.2f}, '
                  f'{time.perf_counter() - started:.0f} s', flush=True)

    reward, cost = (np.array([row[name] for row in rows]) for name in ('reward', 'cost'))
    kinds = sorted(name for name in rows[0] if name.startswith('cost_'))
    summary = {
        'episodes': len(rows),
        'reward': mean_and_error(reward),
        'cost': mean_and_error(cost),
        'episodes within the cost limit': float((cost <= args.limit).mean()),
        'episodes with no cost': float((cost == 0).mean()),
        'highest episode cost': float(cost.max()),
        'share of all cost in the three worst episodes': float(np.sort(cost)[-3:].sum() / cost.sum()) if cost.sum() else None,
        'cost by kind': {name[len('cost_'):]: float(np.mean([row[name] for row in rows])) for name in kinds},
        'decisions at which the action was moved': float(np.mean([row['moved'] for row in rows])),
        'decisions at which a number was moved by more than 0.1': float(np.mean([row['moved by more than 0.1'] for row in rows])),
        'seconds an episode': round((time.perf_counter() - started) / len(rows), 1),
    }
    print(f'\n{player.task}, {described}')
    print(f'reward {summary["reward"][0]:.2f} (standard error {summary["reward"][1]:.2f}), cost {summary["cost"][0]:.2f} '
          f'({summary["cost"][1]:.2f}); within the limit of {args.limit:g} in {summary["episodes within the cost limit"]:.0%} of episodes, '
          f'no cost in {summary["episodes with no cost"]:.0%}, highest {summary["highest episode cost"]:.0f}')
    if 'goals' in rows[0]:
        speeds = np.concatenate([values[1:] for values in by_decision['speed']])
        summary.update({
            'goals reached': mean_and_error([row['goals'] for row in rows]),
            'speed, mean': float(speeds.mean()),
            'speed, the value 10%, 50% and 90% of decisions stay under': [float(v) for v in np.percentile(speeds, [10, 50, 90])]})
        print('cost by kind: ' + ', '.join(f'{name} {value:.2f}' for name, value in summary['cost by kind'].items())
              + f'; goals reached {summary["goals reached"][0]:.1f}; mean speed {summary["speed, mean"]:.3f}')
    if 'success' in rows[0]:
        summary.update({
            'success': mean_and_error([row['success'] for row in rows]),
            'succeeded at some decision': mean_and_error([row['succeeded at some decision'] for row in rows])})
        print(f'success at the end in {summary["success"][0]:.0%} of episodes, at some decision in {summary["succeeded at some decision"][0]:.0%}')

    np.savez_compressed(out / 'scoreboard.npz', seed=np.array(seeds), **{
        name: np.stack(values) for name, values in by_decision.items()})
    (out / 'scoreboard.json').write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'task': player.task,
        'played': described,
        'rendering': os.environ.get('MUJOCO_GL'),
        'first seed': args.first_seed,
        'cost limit': args.limit,
        'summary': summary,
        'each episode': rows,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
