"""Checks imagine.py: that an imagined roll is repeatable, and how far a score from the world model can be trusted.

The adapter is of form residual and size tens throughout.

Two things must hold, or the check fails:

    same again   the same weights, starting states and seed give exactly the same roll twice
    zero         at zero weights no action is changed anywhere in a roll

Measured, with nothing fitted:

    unadapted    the roll at zero weights against what really followed each starting state
                 in its own episode: the cost and the reward in total, imagined over real,
                 and how closely the two follow each other from start to start. For every
                 roll length in --lengths, and for both ways of rolling: next states and
                 actions drawn at random, or the most likely ones.
    seeds        how much the imagined cost of one adapter moves when only the seed of the
                 roll changes
    candidates   21 adapters, each scored in the world model and played in the simulator
                 for --episodes episodes on the same seeds:

                     the dial          steering (see check_adapter.py) with its weights
                                       times 0, 0.1, 0.25, 0.5 and 1. Times 0 is the
                                       unadapted agent.
                     around zero       8 with random weights
                     around steering   8 with steering's weights plus smaller random ones

    agreement    how well the order of the candidates by imagined cost matches their order
                 by real episode cost, and the same for reward (rank correlation: 1 is the
                 same order, 0 is no relation). For every roll length and both ways of
                 rolling, and for reward with and without the value estimate of the last
                 imagined state added. Beside it, how well the real scores agree with
                 themselves when the episodes are split into two halves. A score from the
                 model cannot be expected to agree with the real one better than that.

The starting states come from the last 20 test episodes of the cost predictor, and the
episodes are played on the seeds of the first 20, so the two share no layout of the world.
The cameras are drawn on the GPU, as when the saved episodes were played: nothing here
needs an episode to replay exactly. From this folder:

  ../msCli/ms submit -t 3h -n check-imagine -- apptainer exec --nv --bind /datasets/smohan \\
      /home-mscluster/smohan/images/safeDreamer.sif \\
      python3 check_imagine.py --predictor RUN_FOLDER/cost_predictor.npz --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np
from scipy import stats

from adapter import Adapter
from cost_predictor import files_of, read
from episodes import Player, states
from imagine import Imagination, starting_states
from look_ahead import look_ahead, names, sizes

WAYS = {'drawn': True, 'most likely': False}    # how a roll gets its actions and next states


def same_order(first, second):
    """The rank correlation of two sets of scores: 1 if they put the candidates in the same order."""
    return float(stats.spearmanr(first, second).correlation)


def together(first, second):
    """How closely two sets of numbers follow each other: 1 is exactly, 0 is not at all."""
    return float(np.corrcoef(first, second)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True,
                        help='a cost_predictor.npz, with the cost_predictor.json that was saved beside it')
    parser.add_argument('--decisions', type=int, default=10, help='how many decisions the look-ahead of the adapter lasts')
    parser.add_argument('--push', type=float, default=1.0,
                        help='how far the look-ahead pushes a number towards its limit: 1 is all the way')
    parser.add_argument('--lengths', type=int, nargs='+', default=[10, 20, 40], help='the lengths of imagined roll to try')
    parser.add_argument('--starts', type=int, default=1024, help='how many starting states the rolls have')
    parser.add_argument('--episodes', type=int, default=20, help='episodes played in the simulator with each candidate')
    parser.add_argument('--first-seed', type=int, default=1000, help='the first seed played')
    parser.add_argument('--around', type=int, default=8, help='candidates drawn around zero, and around steering')
    args = parser.parse_args()

    predictor, trained, test_files = read(args.predictor, args.checkpoint)
    if trained['ahead'] != 0:
        raise SystemExit('the look-ahead needs a predictor of the cost of arriving in a state (--ahead 0)')
    for_gpu = {name: predictor[name] for name in ('weights', 'offset', 'mean_cost')}
    lengths = sorted(set(args.lengths))
    longest = lengths[-1]
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    start_files = test_files[len(test_files) // 2:]
    played_on = {f'seed_{seed:04d}.npz' for seed in seeds}
    if played_on & {file.name for file in start_files}:
        raise SystemExit('the seeds to play are also the episodes the starting states come from')

    player = Player(args.checkpoint)
    frozen = player.frozen
    first = np.load(test_files[0])
    action_numbers = first['proposed'].shape[1]
    state_numbers = first['deter'].shape[1] + first['stoch'].shape[1] * int(first['classes'])
    decisions_an_episode = len(first['cost']) - 1

    # The sizes of the look-ahead numbers, from the predictor's training episodes, as in check_adapter.py.
    def look(state, proposed, predictor):
        return look_ahead(frozen, predictor, state, proposed, args.decisions, args.push)

    look = frozen.compiled(look)
    numbers = []
    for file in files_of(trained, 'training'):
        saved = np.load(file)
        numbers.append(look(states(saved), saved['proposed'], for_gpu))
    fixed = {'predictor': for_gpu, 'sizes': sizes(np.concatenate(numbers))}

    adapter = Adapter('residual', 'tens', action_numbers, state_numbers, args.decisions, args.push)
    starts = starting_states(start_files, args.starts, longest)
    imaginations = {way: Imagination(frozen, adapter, fixed, starts['state'], sample) for way, sample in WAYS.items()}
    print(f'{trained["task"]}: {args.starts} starting states from {len(start_files)} episodes ({start_files[0].name} to '
          f'{start_files[-1].name}), {starts["charged"].mean():.1%} of them charged. Look-ahead of {args.decisions} decisions.',
          flush=True)

    # The candidates. Steering is written down as in check_adapter.py: a table with a line per
    # output, the gains first and then the shifts, and a column per look-ahead number.
    called = names(action_numbers)
    steering = np.zeros((2 * action_numbers, len(called) + 1), np.float32)
    for number in range(action_numbers):
        steering[action_numbers + number, called.index(f'cost change, number {number} up')] = -1
        steering[action_numbers + number, called.index(f'cost change, number {number} down')] = 1
    steering = steering.ravel()
    random = np.random.default_rng(0)
    candidates, group = {}, {}
    for scale in (0, 0.1, 0.25, 0.5, 1):
        candidates[f'steering x {scale:g}'] = (scale * steering).astype(np.float32)
        group[f'steering x {scale:g}'] = 'the dial'
    for number in range(args.around):
        candidates[f'around zero {number}'] = random.normal(0, 0.3, adapter.count).astype(np.float32)
        group[f'around zero {number}'] = 'around zero'
    for number in range(args.around):
        candidates[f'around steering {number}'] = (steering + random.normal(0, 0.1, adapter.count)).astype(np.float32)
        group[f'around steering {number}'] = 'around steering'
    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    np.savez(out / 'candidates.npz', **candidates)

    # same again and zero.
    once = imaginations['drawn'].roll(candidates['steering x 1'], longest)
    started = time.perf_counter()
    again = imaginations['drawn'].roll(candidates['steering x 1'], longest)
    seconds_a_roll = time.perf_counter() - started
    verdicts = {'same again': all(np.array_equal(once[name], again[name]) for name in once)}
    rolls = {way: {name: imagination.roll(weights, longest) for name, weights in candidates.items()}
             for way, imagination in imaginations.items()}
    verdicts['zero'] = not any(rolls[way]['steering x 0']['moved'].any() for way in WAYS)
    print(f'a roll of {longest} decisions from {args.starts} starting states takes {seconds_a_roll:.2f} s', flush=True)

    # unadapted: the roll at zero weights against what really followed the same starting states.
    free = ~starts['charged']
    unadapted = {}
    for way in WAYS:
        roll = rolls[way]['steering x 0']
        for length in lengths:
            imagined_cost, followed_cost = roll['cost'][:length].sum(0), starts['cost'][:, :length].sum(1)
            imagined_reward, followed_reward = roll['reward'][:length].sum(0), starts['reward'][:, :length].sum(1)
            unadapted[f'{way}, {length}'] = {
                'cost, imagined over real': float(imagined_cost.sum() / followed_cost.sum()),
                'cost, imagined over real, starts not charged': float(imagined_cost[free].sum() / followed_cost[free].sum()),
                'cost, imagined with real': together(imagined_cost, followed_cost),
                'reward, imagined over real': float(imagined_reward.sum() / followed_reward.sum()),
                'reward, imagined with real': together(imagined_reward, followed_reward),
            }
    print('unadapted, the roll at zero weights against what really followed (imagined over real; in brackets, how closely '
          'they follow each other from start to start):')
    for name, found in unadapted.items():
        print(f'  {name:15s} cost {found["cost, imagined over real"]:.2f} ({found["cost, imagined with real"]:.2f}), '
              f'from starts not charged {found["cost, imagined over real, starts not charged"]:.2f}; '
              f'reward {found["reward, imagined over real"]:.2f} ({found["reward, imagined with real"]:.2f})')

    # seeds: the same adapter and starting states, with other random numbers.
    def episode_cost(roll, length):
        """The cost of an episode if every decision cost what the first `length` of the roll cost on average."""
        return float(roll['cost'][:length].mean() * decisions_an_episode)

    def episode_reward(roll, length):
        return float(roll['reward'][:length].mean() * decisions_an_episode)

    by_seed = {name: [episode_cost(imaginations['drawn'].roll(candidates[name], longest, seed), longest) for seed in range(5)]
               for name in ('steering x 0', 'steering x 1')}
    print('seeds, imagined episode cost with five seeds: '
          + '; '.join(f'{name}: ' + ', '.join(f'{cost:.1f}' for cost in costs) for name, costs in by_seed.items()), flush=True)

    # candidates: each one played in the simulator.
    act = adapter.compiled(frozen)
    real = {}
    for name, weights in candidates.items():
        started = time.perf_counter()
        if name == 'steering x 0':
            episodes = list(player.play(seeds))
        else:
            episodes = list(player.play(
                seeds, adapter=lambda state, proposed, weights=weights: act(state, proposed, fixed, weights)))
        real[name] = {'reward': np.array([episode['reward'].sum() for episode in episodes], np.float64),
                      'cost': np.array([episode['cost'].sum() for episode in episodes], np.float64)}
        print(f'  played {name:19s} reward {real[name]["reward"].mean():6.2f}, cost {real[name]["cost"].mean():6.1f} '
              f'({time.perf_counter() - started:.0f} s)', flush=True)

    # agreement: the order of the candidates by imagined score against their order by real score.
    order = list(candidates)
    real_cost = np.array([real[name]['cost'].mean() for name in order])
    real_reward = np.array([real[name]['reward'].mean() for name in order])
    groups = {'all': np.ones(len(order), bool)}
    for kind in ('the dial', 'around zero', 'around steering'):
        groups[kind] = np.array([group[name] == kind for name in order])
    halves = [np.arange(args.episodes) % 2 == half for half in (0, 1)]
    ceiling = {kind: {
        'cost': same_order(*[np.array([real[name]['cost'][half].mean() for name in order])[members] for half in halves]),
        'reward': same_order(*[np.array([real[name]['reward'][half].mean() for name in order])[members] for half in halves]),
    } for kind, members in groups.items()}
    imagined, agreement = {}, {}
    for way in WAYS:
        for length in lengths:
            scores = {
                'cost': np.array([episode_cost(rolls[way][name], length) for name in order]),
                'reward': np.array([episode_reward(rolls[way][name], length) for name in order]),
                # The reward of the roll with the value estimate of its last state, per starting state.
                'reward with value': np.array([float((rolls[way][name]['reward'][:length].sum(0)
                                                      + rolls[way][name]['value'][length - 1]).mean()) for name in order]),
            }
            imagined[f'{way}, {length}'] = scores
            agreement[f'{way}, {length}'] = {kind: {
                'cost': same_order(scores['cost'][members], real_cost[members]),
                'reward': same_order(scores['reward'][members], real_reward[members]),
                'reward with value': same_order(scores['reward with value'][members], real_reward[members]),
            } for kind, members in groups.items()}

    print(f'\nagreement of the order by imagined score with the order by real score, over {len(order)} candidates '
          f'and {args.episodes} episodes each (all / the dial / around zero / around steering):')
    for name, found in agreement.items():
        print(f'  {name:15s} ' + '; '.join(
            f'{score} ' + ' / '.join(f'{found[kind][score]:+.2f}' for kind in groups)
            for score in ('cost', 'reward', 'reward with value')))
    print(f'  {"real, two halves":15s} ' + '; '.join(
        f'{score} ' + ' / '.join(f'{ceiling[kind][score]:+.2f}' for kind in groups) for score in ('cost', 'reward')))

    shown = f'drawn, {longest}'
    print(f'\ncandidates, real against imagined ({shown}): reward, cost')
    for position, name in enumerate(order):
        print(f'  {name:19s} real {real_reward[position]:6.2f}, {real_cost[position]:6.1f}; imagined '
              f'{imagined[shown]["reward"][position]:6.2f}, {imagined[shown]["cost"][position]:6.1f}; '
              f'action changed at {rolls["drawn"][name]["moved"].mean():.0%} of imagined decisions')
    for name, verdict in verdicts.items():
        print(f'{name}: {verdict}')

    (out / 'check_imagine.json').write_text(json.dumps({
        'checkpoint': trained['checkpoint'],
        'task': trained['task'],
        'predictor': args.predictor,
        'adapter': adapter.describe(),
        'rendering': os.environ.get('MUJOCO_GL'),
        'starting states': {'count': args.starts, 'from': [start_files[0].name, start_files[-1].name, len(start_files)],
                            'charged': float(starts['charged'].mean())},
        'seeds played': seeds,
        'verdicts': verdicts,
        'unadapted, the roll at zero weights against what really followed': unadapted,
        'seeds, imagined episode cost': by_seed,
        'candidates': {name: {
            'group': group[name],
            'real reward': float(real[name]['reward'].mean()),
            'real cost': float(real[name]['cost'].mean()),
            'real reward, each episode': [float(value) for value in real[name]['reward']],
            'real cost, each episode': [float(value) for value in real[name]['cost']],
            'imagined': {setting: {score: float(values[position]) for score, values in scores.items()}
                         for setting, scores in imagined.items()},
            'imagined decisions at which the action was changed': float(rolls['drawn'][name]['moved'].mean()),
        } for position, name in enumerate(order)},
        'agreement': agreement,
        'agreement of the real scores of two halves of the episodes': ceiling,
    }, indent=2) + '\n')
    if not all(verdicts.values()):
        raise SystemExit('FAIL')
    print('PASS: the two exact checks hold; the rest above is measured, not judged')


if __name__ == '__main__':
    main()
