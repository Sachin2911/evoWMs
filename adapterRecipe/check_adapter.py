"""Checks adapter.py: that an adapter at zero weights changes nothing, and what each form does before any search.

Seven kinds of adapter are checked: a residual and a mix of each size, and the re-rank.

Six things must hold, or the check fails:

    weights        every kind has the number of weights the design gives it, and refuses
                   any other number
    zero           with every weight at zero, every kind gives back exactly the proposed
                   action, at every decision of the test episodes
    zero, played   an episode played in the simulator with a kind at zero weights is the
                   unadapted agent's episode: every action, reward and cost is the same
    finite         with random weights every action is a finite number, and an action
                   number that a residual changed lies within the limits
    same again     the same states, weights and random numbers give exactly the same
                   actions a second time
    told           in every episode played with an adapter, the world model was told the
                   action that was taken: one roll forward from a saved state with the
                   saved action gives the continuous part of the next saved state

Measured, with nothing fitted:

    sizes         what the look-ahead numbers are divided by, from the training episodes
                  of the cost predictor
    candidates    over the decisions of the test episodes that are not charged and where
                  holding the proposed action is expected to cost half a unit or more: how
                  often some candidate at least halves that. For the four pushes of the
                  look-ahead, and for 4, 16 and 64 further actions from the agent. This is
                  what a re-rank has to work with (decision 3 of the design).
    time          milliseconds for one decision, for every kind
    played        whole episodes in the simulator with three adapters whose weights are
                  written down by hand, against the unadapted agent on the same seeds:

                      re-rank, cost only    the candidate expected to cost least
                      residual, steering    size tens; each action number is shifted
                                            towards the push expected to cost less, by the
                                            difference of the two cost changes
                      mix, both             the two together

                  None of the three is searched or tuned. They show that an adapter with
                  weights other than zero runs from end to end, and give a first look at
                  what the forms do in real episodes.

Run it with the cameras drawn on the CPU (MUJOCO_GL=osmesa), as check_frozen_agent.py is.
From this folder:

  ../msCli/ms submit -t 2h -n check-adapter -- apptainer exec --nv --bind /datasets/smohan \\
      --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \\
      python3 check_adapter.py --predictor RUN_FOLDER/cost_predictor.npz --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np

from adapter import Adapter
from cost_predictor import files_of, read
from episodes import Player, states
from look_ahead import look_ahead, names, sizes


def mean_and_range(values, draws=1000):
    """The mean, and the range that holds 95% of the means when the episodes are redrawn with replacement."""
    values = np.asarray(values, np.float64)
    random = np.random.default_rng(0)
    means = values[random.integers(len(values), size=(draws, len(values)))].mean(1)
    return [float(values.mean())] + [float(end) for end in np.percentile(means, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True,
                        help='a cost_predictor.npz, with the cost_predictor.json that was saved beside it')
    parser.add_argument('--decisions', type=int, default=10, help='how many decisions the imagined futures last')
    parser.add_argument('--push', type=float, default=1.0,
                        help='how far the look-ahead pushes a number towards its limit: 1 is all the way')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes played with each hand-written adapter and with none')
    parser.add_argument('--first-seed', type=int, default=1000,
                        help='the first seed played; 1000 to 1039 are the test episodes of the cost predictor')
    args = parser.parse_args()

    # `files` are the episodes the predictor was scored on, which it has never been fitted to.
    predictor, trained, files = read(args.predictor, args.checkpoint)
    if trained['ahead'] != 0:
        raise SystemExit('the look-ahead needs a predictor of the cost of arriving in a state (--ahead 0)')
    for_gpu = {name: predictor[name] for name in ('weights', 'offset', 'mean_cost')}
    player = Player(args.checkpoint)
    frozen = player.frozen
    first = np.load(files[0])
    action_numbers = first['proposed'].shape[1]
    state_numbers = first['deter'].shape[1] + first['stoch'].shape[1] * int(first['classes'])

    def look(state, proposed, predictor):
        return look_ahead(frozen, predictor, state, proposed, args.decisions, args.push)

    def roll(state, action):
        return frozen.roll_forward(state, action)['deter']

    look, roll = frozen.compiled(look), frozen.compiled(roll)

    # The sizes of the look-ahead numbers, over every decision of the predictor's training episodes.
    training = files_of(trained, 'training')
    numbers = []
    for file in training:
        saved = np.load(file)
        numbers.append(look(states(saved), saved['proposed'], for_gpu))
    measured = sizes(np.concatenate(numbers))
    fixed = {'predictor': for_gpu, 'sizes': measured}
    called = names(action_numbers)
    print(f'{trained["task"]}, futures of {args.decisions} decisions, push {args.push:g}. Sizes from {len(training)} training episodes: '
          f'{called[0]} {measured[0]:.3f}, a cost change {measured[1]:.3f}, a reward change {measured[-1]:.3f}', flush=True)

    def make(form, size, **settings):
        return Adapter(form, size, action_numbers, state_numbers, args.decisions, args.push, **settings)

    kinds = {'re-rank': make('re-rank', None)}
    for size in ('tens', 'hundreds', 'thousands'):
        kinds[f'residual, {size}'] = make('residual', size)
        kinds[f'mix, {size}'] = make('mix', size)
    acts = {kind: adapter.compiled(frozen) for kind, adapter in kinds.items()}
    verdicts = {}

    # weights: the numbers in the design, written out again here, and a refusal of any other number.
    look_ahead_numbers = 4 * action_numbers + 1
    outputs = 2 * action_numbers
    designed = {'tens': (look_ahead_numbers + 1) * outputs,
                'hundreds': 16 * (look_ahead_numbers + 1) + outputs * 17,
                'thousands': state_numbers + outputs + outputs}
    one_state = ({name: part[:1] for name, part in states(first).items()}, first['proposed'][:1])
    right_count, refuses = True, True
    for kind, adapter in kinds.items():
        form, _, size = kind.partition(', ')
        right_count &= adapter.count == (designed[size] if size else 0) + (0 if form == 'residual' else 2)
        for wrong in (adapter.count - 1, adapter.count + 1):
            try:
                adapter.compiled(frozen)(*one_state, fixed, np.zeros(wrong, np.float32))
                refuses = False
            except ValueError:
                pass
    verdicts['weights'] = bool(right_count and refuses)
    print('weights: ' + ', '.join(f'{kind} {adapter.count}' for kind, adapter in kinds.items()), flush=True)

    # zero, finite and same again, on every decision of the test episodes.
    random = np.random.default_rng(0)
    # Random weights of a size at which every kind changes a good share of the actions.
    scattered = {kind: random.normal(0, 0.3, adapter.count).astype(np.float32) for kind, adapter in kinds.items()}
    zero_changes = {kind: 0 for kind in kinds}
    all_finite, within_limits, same_again = True, True, True
    changed = {kind: [] for kind in kinds}
    for number, file in enumerate(files):
        saved = np.load(file)
        state, proposed = states(saved), saved['proposed']
        held = np.clip(proposed, -1, 1)
        for kind, adapter in kinds.items():
            at_zero = acts[kind](state, proposed, fixed, np.zeros(adapter.count, np.float32))
            zero_changes[kind] += int((at_zero != proposed).sum())
            # The random numbers of an adapter are set by the episode's seed when playing, and by hand here.
            frozen.reseed(number)
            action = acts[kind](state, proposed, fixed, scattered[kind])
            frozen.reseed(number)
            same_again &= np.array_equal(action, acts[kind](state, proposed, fixed, scattered[kind]))
            all_finite &= bool(np.isfinite(action).all())
            moved = action != proposed
            if kind.startswith('residual'):
                within_limits &= bool((np.abs(action[moved]) <= 1).all())
            changed[kind].append((np.clip(action, -1, 1) != held).any(1))
    verdicts['zero'] = all(count == 0 for count in zero_changes.values())
    verdicts['finite'] = bool(all_finite and within_limits)
    verdicts['same again'] = bool(same_again)
    print(f'zero: action numbers changed at zero weights, over {len(files)} test episodes: '
          + ', '.join(f'{kind} {count}' for kind, count in zero_changes.items()))
    print('random weights: share of decisions at which the action taken changes: '
          + ', '.join(f'{kind} {np.concatenate(shares).mean():.0%}' for kind, shares in changed.items()), flush=True)

    # candidates: what a re-rank has to choose from where the look-ahead expects a cost.
    many = (4, 16, 64)
    wide = make('re-rank', None, draws=many[-1]).compiled(frozen, everything=True)
    halved = {'pushes': []}
    halved.update({f'{count} further actions': [] for count in many})
    lowest = {name: [] for name in halved}
    apart, expecting_count, free_count = [], 0, 0
    for number, file in enumerate(files):
        saved = np.load(file)
        state, proposed = states(saved), saved['proposed']
        frozen.reseed(number)
        ranked = wide(state, proposed, fixed, np.zeros(2, np.float32))
        pushes = look(state, proposed, for_gpu)[:, 1:1 + 2 * action_numbers]
        holding = ranked['cost'][0]
        # Not charged, and not the entry that ends the episode, where no action is taken.
        free = saved['cost'] == 0
        free[-1] = False
        expecting = free & (holding >= 0.5)
        free_count += int(free.sum())
        expecting_count += int(expecting.sum())
        # The lowest cost each set of candidates offers, as a share of the cost of holding.
        offered = {'pushes': 1 + pushes[expecting].min(1) / holding[expecting]}
        for count in many:
            offered[f'{count} further actions'] = ranked['cost'][1:1 + count][:, expecting].min(0) / holding[expecting]
        for name, share in offered.items():
            halved[name].append(share <= 0.5)
            lowest[name].append(share)
        clipped = np.clip(ranked['candidates'], -1, 1)
        apart.append(np.abs(clipped[1:] - clipped[:1])[:, expecting].mean(0))
    candidates = {name: {'some candidate at least halves it': float(np.concatenate(halved[name]).mean()),
                         'lowest cost on offer over the cost of holding, middle': float(np.median(np.concatenate(lowest[name])))}
                  for name in halved}
    apart = np.concatenate(apart).mean(0)
    print(f'candidates: {expecting_count} of {free_count} decisions that are not charged expect a cost of 0.5 or more from holding')
    for name, found in candidates.items():
        print(f'  {name:19s} some candidate at least halves it at {found["some candidate at least halves it"]:4.0%}; '
              f'lowest cost on offer, middle: {found["lowest cost on offer over the cost of holding, middle"]:.2f} of holding')
    print('  a further action lies this far from the proposed one, mean per action number: '
          + ', '.join(f'{distance:.2f}' for distance in apart) + ' (a push all the way moves a number by up to 2)', flush=True)

    # time: one state at a time, as when playing. The first call for a new shape also compiles.
    milliseconds = {}
    singles = [({name: part[entry:entry + 1] for name, part in states(first).items()}, first['proposed'][entry:entry + 1])
               for entry in range(101)]
    for kind in kinds:
        acts[kind](*singles[0], fixed, scattered[kind])
        started = time.perf_counter()
        for single in singles[1:]:
            acts[kind](*single, fixed, scattered[kind])
        milliseconds[kind] = (time.perf_counter() - started) * 1000 / 100
    print('time for one decision, ms: ' + ', '.join(f'{kind} {took:.1f}' for kind, took in milliseconds.items()), flush=True)

    # From here on the simulator runs.
    def with_weights(kind, weights):
        return lambda state, proposed: acts[kind](state, proposed, fixed, weights)

    def same(first_run, second_run):
        return all(np.array_equal(one[name], other[name])
                   for one, other in zip(first_run, second_run) for name in ('action', 'reward', 'cost'))

    def told(episode):
        """The mean gap between one roll forward from each saved state and the next saved state."""
        state = states(episode, slice(0, -1))
        return float(np.abs(roll(state, episode['action'][:-1]).astype(np.float32)
                            - episode['deter'][1:].astype(np.float32)).mean())

    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    started = time.perf_counter()
    unadapted = list(player.play(seeds))
    print(f'unadapted: {len(seeds)} episodes played in {time.perf_counter() - started:.0f} s', flush=True)

    # zero, played: two episodes with each kind at zero weights.
    replayed = {}
    for kind, adapter in kinds.items():
        played = list(player.play(seeds[:2], adapter=with_weights(kind, np.zeros(adapter.count, np.float32))))
        replayed[kind] = same(unadapted, played)
    print('zero, played: ' + ', '.join(f'{kind} {verdict}' for kind, verdict in replayed.items()), flush=True)

    # played: three adapters whose weights are written down by hand.
    # The table of size tens: a line per output, the gains first and then the shifts, and a
    # column per look-ahead number with one more that is added on (see adapter.py).
    steering = np.zeros((outputs, look_ahead_numbers + 1), np.float32)
    for number in range(action_numbers):
        steering[action_numbers + number, called.index(f'cost change, number {number} up')] = -1
        steering[action_numbers + number, called.index(f'cost change, number {number} down')] = 1
    cost_only = np.array([0, 1], np.float32)    # reward weight, cost weight
    by_hand = {
        're-rank, cost only': ('re-rank', cost_only),
        'residual, steering': ('residual, tens', steering.ravel()),
        'mix, both': ('mix, tens', np.concatenate([steering.ravel(), cost_only])),
    }

    def totals(episodes):
        return (np.array([episode['reward'].sum() for episode in episodes], np.float64),
                np.array([episode['cost'].sum() for episode in episodes], np.float64))

    reward_alone, cost_alone = totals(unadapted)
    gaps = {'unadapted': [told(episode) for episode in unadapted]}
    report = {'unadapted': {'reward': mean_and_range(reward_alone), 'cost': mean_and_range(cost_alone)}}
    print(f'played, {len(seeds)} episodes each, mean (95% range):')
    print(f'  {"unadapted":19s} reward {reward_alone.mean():6.2f}, cost {cost_alone.mean():6.1f}')
    for name, (kind, weights) in by_hand.items():
        started = time.perf_counter()
        played = list(player.play(seeds, adapter=with_weights(kind, weights)))
        reward, cost = totals(played)
        gaps[name] = [told(episode) for episode in played]
        # How far each action number was moved at each decision, as the simulator takes the actions.
        moves = np.concatenate([np.abs(np.clip(episode['action'], -1, 1) - np.clip(episode['proposed'], -1, 1))[:-1]
                                for episode in played])
        report[name] = {
            'kind': kinds[kind].describe(),
            'reward': mean_and_range(reward),
            'cost': mean_and_range(cost),
            'reward change': mean_and_range(reward - reward_alone),
            'cost change': mean_and_range(cost - cost_alone),
            'reward over the unadapted agent\'s': float(reward.mean() / reward_alone.mean()),
            'cost over the unadapted agent\'s': float(cost.mean() / cost_alone.mean()),
            'episodes with less cost, the same, more': [int((cost < cost_alone).sum()), int((cost == cost_alone).sum()),
                                                        int((cost > cost_alone).sum())],
            'decisions at which the action was changed at all': float((moves > 0).any(1).mean()),
            'decisions at which a number was moved by more than 0.1': float((moves > 0.1).any(1).mean()),
            'mean move of each action number': [float(move) for move in moves.mean(0)],
            'seconds an episode': round((time.perf_counter() - started) / len(seeds), 1),
            'episodes': [{'seed': seed, 'reward': float(one), 'cost': float(other)}
                         for seed, one, other in zip(seeds, reward, cost)],
        }
        found = report[name]
        print(f'  {name:19s} reward {reward.mean():6.2f}, cost {cost.mean():6.1f}; against unadapted: reward '
              f'{found["reward change"][0]:+.2f} ({found["reward change"][1]:+.2f} to {found["reward change"][2]:+.2f}), cost '
              f'{found["cost change"][0]:+.1f} ({found["cost change"][1]:+.1f} to {found["cost change"][2]:+.1f}); '
              f'less cost in {found["episodes with less cost, the same, more"][0]} episodes, more in '
              f'{found["episodes with less cost, the same, more"][2]}; a number moved by more than 0.1 at '
              f'{found["decisions at which a number was moved by more than 0.1"]:.0%} of decisions, changed at all at '
              f'{found["decisions at which the action was changed at all"]:.0%}', flush=True)
    report['unadapted']['episodes'] = [{'seed': seed, 'reward': float(one), 'cost': float(other)}
                                       for seed, one, other in zip(seeds, reward_alone, cost_alone)]

    # The unadapted agent once more, to know that episodes replay in this job at all.
    replays = same(unadapted, list(player.play(seeds[:2])))
    verdicts['zero, played'] = all(replayed.values())
    largest_gap = max(max(found) for found in gaps.values())
    verdicts['told'] = largest_gap < 1e-3
    print(f'told: the largest mean gap of an episode is {largest_gap:.2g} '
          f'(unadapted episodes: {max(gaps["unadapted"]):.2g})')
    for name, verdict in verdicts.items():
        print(f'{name}: {verdict}')

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'check_adapter.json'
    out.write_text(json.dumps({
        'checkpoint': trained['checkpoint'],
        'task': trained['task'],
        'predictor': args.predictor,
        'decisions': args.decisions,
        'push': args.push,
        'rendering': os.environ.get('MUJOCO_GL'),
        'verdicts': verdicts,
        'episodes replay at all': bool(replays),
        'sizes': {'from': trained['files']['training'], called[0]: float(measured[0]),
                  'a cost change': float(measured[1]), 'a reward change': float(measured[-1])},
        'kinds': {kind: adapter.describe() for kind, adapter in kinds.items()},
        'zero': {'action numbers changed': zero_changes, 'played': replayed},
        'random weights, decisions at which the action taken changes':
            {kind: float(np.concatenate(shares).mean()) for kind, shares in changed.items()},
        'candidates': {'decisions not charged': free_count, 'of those, expecting 0.5 or more from holding': expecting_count,
                       'sets': candidates, 'distance of a further action from the proposed one': [float(d) for d in apart]},
        'milliseconds for one decision': milliseconds,
        'seeds played': seeds,
        'played': report,
        'told, mean gap of each episode': gaps,
    }, indent=2) + '\n')

    if not replays:
        raise SystemExit('INCONCLUSIVE: the unadapted agent did not replay itself in this job')
    if not all(verdicts.values()):
        raise SystemExit('FAIL')
    print('PASS: the six exact checks hold; the rest above is measured, not judged')


if __name__ == '__main__':
    main()
