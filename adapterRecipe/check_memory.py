"""Checks the adapter with memory on the real frozen agent, before any search.

test_memory.py checks the arithmetic with a made-up agent. This checks what only the real
agent and its simulator can show. These must hold, or the check fails:

    zero           with every weight at zero the adapter gives back exactly the proposed
                   action at every decision of the test episodes, and its memory stays zero
    zero, played   an episode played in the simulator with the adapter at zero weights is
                   the unadapted agent's episode: every action, reward and cost is the same
    bounded        with random weights every output and every memory lies between -1 and 1,
                   and every action number the adapter changed lies within the limits
                   (to the last digit a GPU gives)
    same again     the same states, weights and memory give exactly the same actions a
                   second time
    remembered     in an episode played with random weights, the memory saved at each
                   decision is the one that the decision before handed on, exactly: it is
                   carried from one decision to the next and never worked out twice
    told           in that episode the world model was told the action that was taken (for
                   DreamerV3: one roll forward from a saved state with the saved action
                   gives the continuous part of the next saved state)
    value          for TD-MPC2: the value the adapter reads, which leaves out the dropout
                   the agent's Q-networks play with, lies as near the mean of 32 values
                   with the dropout as one of those does

Measured, with nothing fitted:

    sizes          what the look-ahead numbers are divided by, by stretch
    time           milliseconds for one decision
    worked out again   how far the adapter's outputs for the decisions of a played episode
                   lie from the same decisions worked out again: one at a time, and all
                   together as an imagined roll works them out. The agent computes in
                   half the digits, the last of them depend on how a computation is laid
                   out, and a look-ahead that takes the most likely class at every step
                   jumps when two classes are nearly as likely. This is measured with
                   random weights and with steering
    played         whole episodes with steering written in this form, against the
                   unadapted agent on the same seeds

From this folder:

  ../msCli/ms submit -t 1h -n check-memory -- apptainer exec --nv --bind /datasets/smohan \\
      /home-mscluster/smohan/images/safeDreamer.sif python3 check_memory.py \\
      --predictor RUN_FOLDER/cost_predictor.npz --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np

from adapter import MemoryAdapter
from cost_predictor import files_of, read
from look_ahead import measured_by_stretch
from search import player_of


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True)
    parser.add_argument('--episodes', type=int, default=3, help='episodes played for each of: unadapted, zero weights, steering')
    parser.add_argument('--first-seed', type=int, default=3000)
    parser.add_argument('--look', default='likely', choices=('likely', 'mean'), help='what the look-ahead rolls on')
    args = parser.parse_args()

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    predictor, trained, test_files = read(args.predictor, args.checkpoint)
    for_gpu = {name: predictor[name] for name in predictor if name != 'ahead'}
    player = player_of(args.checkpoint)
    frozen = player.frozen
    adapter = MemoryAdapter(player.action_numbers, roll=args.look)
    fixed = {'predictor': for_gpu, 'sizes': measured_by_stretch(frozen, for_gpu, files_of(trained, 'training'),
                                                                adapter.lengths, adapter.stop, adapter.roll)}
    act = adapter.compiled(frozen, everything=True)
    zero = np.zeros(adapter.count, np.float32)
    random = np.random.default_rng(0)
    result, failed = {'task': player.task, 'adapter': adapter.describe(),
                      'sizes': dict(zip(adapter.names(), (float(size) for size in fixed['sizes'])))}, []

    def held_to(name, holds, said):
        result[name] = said
        print(f'{"ok    " if holds else "FAILED"} {name}: {said}', flush=True)
        if not holds:
            failed.append(name)

    # zero, bounded and same again, on the states of two test episodes.
    saved = [np.load(file) for file in test_files[:2]]
    states = [frozen.written_out(frozen.kept(episode)) for episode in saved]
    decisions = sum(len(episode['proposed']) for episode in saved)
    same = all(np.array_equal(made['action'], episode['proposed']) and not made['memory'].any() and not made['output'].any()
               for episode, state in zip(saved, states)
               for made in [act(state, episode['proposed'], adapter.start(len(episode['proposed'])), fixed, zero)])
    held_to('zero', same, f'the proposed action came back at all {decisions} decisions, with the memory at zero')

    weights = random.normal(0, 1, adapter.count).astype(np.float32)
    memory = random.uniform(-1, 1, (len(saved[0]['proposed']), adapter.remembered)).astype(np.float32)
    once = act(states[0], saved[0]['proposed'], memory, fixed, weights)
    again = act(states[0], saved[0]['proposed'], memory, fixed, weights)
    changed = once['action'] != saved[0]['proposed']
    # A tanh worked out on a GPU can come out one last digit above 1, so the bound has that much room.
    most = 1 + 1e-5
    bounded = (np.isfinite(once['action']).all() and (np.abs(once['output']) <= most).all()
               and (np.abs(once['memory']) <= most).all() and (np.abs(once['action'][changed]) <= most).all())
    held_to('bounded', bool(bounded), f'{changed.any(1).mean():.0%} of decisions changed with random weights; '
            f'the largest output is {np.abs(once["output"]).max():.3f} and the largest memory {np.abs(once["memory"]).max():.3f}')
    held_to('same again', all(np.array_equal(once[name], again[name]) for name in once), 'the same actions, outputs and memories twice')

    if hasattr(frozen, 'worth_as_played'):
        # The value is read without the dropout the agent's Q-networks play with. It has to be the same networks.
        quiet = frozen.compiled(frozen.worth)(states[0])
        as_played = np.stack([frozen.compiled(frozen.worth_as_played)(states[0]) for _ in range(32)])
        gap, spread = np.median(np.abs(quiet - as_played.mean(0))), np.median(as_played.std(0))
        held_to('value', bool(gap <= spread), f'without the dropout of the Q-networks the value is {gap:.3f} from the mean of 32 values '
                f'with it, at the median state; one value with it is {spread:.3f} from that mean')

    one = {name: part[:1] for name, part in states[0].items()}
    act(one, saved[0]['proposed'][:1], adapter.start(), fixed, weights)     # the first call for one state is slower
    started = time.perf_counter()
    for _ in range(20):
        act(one, saved[0]['proposed'][:1], adapter.start(), fixed, weights)
    result['milliseconds a decision'] = round((time.perf_counter() - started) / 20 * 1000, 1)
    print(f'       time: {result["milliseconds a decision"]} ms for one decision', flush=True)

    # zero, played: the same seeds with no adapter and with the adapter at zero weights.
    def with_weights(weights):
        compiled = adapter.compiled(frozen)
        return lambda state, proposed, held: compiled(state, proposed, held, fixed, weights)

    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    unadapted = list(player.play(seeds))
    at_zero = list(player.play(seeds, adapter=with_weights(zero), memory=adapter.start()))
    same = all(np.array_equal(one[name], other[name]) for one, other in zip(unadapted, at_zero) for name in ('action', 'reward', 'cost'))
    held_to('zero, played', same, f'{len(seeds)} episodes with the adapter at zero weights are the unadapted agent\'s, action for action')

    # remembered and told, in one episode played with small random weights.
    small = (0.3 * weights).astype(np.float32)
    with_small, calls = with_weights(small), []

    def recording(state, proposed, held):
        """The adapter, with what it was handed and what it gave kept for every decision."""
        action, following = with_small(state, proposed, held)
        calls.append((held, action, following))
        return action, following

    episode, = player.play([args.first_seed], adapter=recording, memory=adapter.start())
    entries = len(episode['proposed']) - 1     # the last entry's action is never taken
    carried = all(np.array_equal(calls[entry + 1][0], calls[entry][2]) for entry in range(len(calls) - 1))
    # An episode has one more entry than it has decisions. A player that does not ask the
    # adapter at the last entry saves there the memory the last decision handed on.
    read = np.concatenate([held for held, _, _ in calls])
    saved_as_read = (np.array_equal(episode['memory'][:len(read)], read)
                     and all(np.array_equal(row[None], calls[-1][2]) for row in episode['memory'][len(read):]))
    # The episode keeps an action in the agent's half digits, so the last digits of the two can differ.
    action_gap = float(np.abs(np.concatenate([action for _, action, _ in calls])[:entries] - episode['action'][:entries]).max())
    held_to('remembered', carried and saved_as_read and action_gap < 1e-3 and not episode['memory'][0].any(),
            f'over {len(calls)} decisions each memory read is the one the decision before handed on, the episode saved them as '
            f'read, and the actions saved are the adapter\'s to within {action_gap:.5f}; the memory reaches '
            f'{np.abs(episode["memory"]).max():.3f}')

    # worked out again: the played decisions one at a time and as one batch.
    state = frozen.written_out(frozen.kept(episode, slice(0, entries)))

    def one_at_a_time(weights, memories):
        """What the adapter gives for each saved decision on its own, the way an episode is played."""
        each = [act({name: part[entry:entry + 1] for name, part in state.items()}, episode['proposed'][entry:entry + 1],
                    memories[entry:entry + 1], fixed, weights) for entry in range(entries)]
        return {name: np.concatenate([made[name] for made in each]) for name in each[0]}

    def spread(gap):
        return {'median': float(np.median(gap)), '9 in 10 within': float(np.quantile(gap, 0.9)),
                '99 in 100 within': float(np.quantile(gap, 0.99)), 'largest': float(gap.max()),
                'share above 0.05': float((gap > 0.05).mean())}

    def said(gap):
        return (f'{np.median(gap):.4f} at the median decision, less than {np.quantile(gap, 0.9):.4f} at 9 in 10 and '
                f'{np.quantile(gap, 0.99):.4f} at 99 in 100, {gap.max():.4f} at the most; {(gap > 0.05).mean():.1%} above 0.05')

    result['worked out again'] = {}
    for name, tried, memories in (('random weights', small, episode['memory']),
                                  ('steering', adapter.steering(), np.zeros_like(episode['memory']))):
        alone = one_at_a_time(tried, memories)
        together = act(state, episode['proposed'][:entries], memories[:entries], fixed, tried)
        gap = np.abs(together['output'] - alone['output']).max(1)
        result['worked out again'][name] = {'together against one at a time, outputs': spread(gap)}
        print(f'       worked out again, {name}: all together against one at a time, the outputs differ by {said(gap)}', flush=True)
        if name == 'random weights':
            gap = np.abs(alone['action'] - episode['action'][:entries]).max(1)
            result['worked out again'][name]['one at a time against the episode, actions'] = spread(gap)
            print(f'       worked out again, {name}: one at a time against the episode as played, the actions differ by {said(gap)}',
                  flush=True)
    if hasattr(frozen, 'roll_forward'):
        def roll(state, action):
            return frozen.roll_forward(state, action, sample=False)['deter']

        expected = frozen.compiled(roll)(state, episode['action'][:entries])
        gap = float(np.abs(expected.astype(np.float32) - episode['deter'][1:entries + 1].astype(np.float32)).max())
        held_to('told', gap < 1e-2, f'one roll forward with the saved action gives the next saved state to within {gap:.4f}')

    # played: steering written in this form, against the unadapted agent on the same seeds.
    steering = list(player.play(seeds, adapter=with_weights(adapter.steering()), memory=adapter.start()))
    result['played'] = {
        name: {'reward': [float(episode['reward'].sum()) for episode in episodes],
               'cost': [float(episode['cost'].sum()) for episode in episodes],
               'seconds an episode': float(np.mean([episode['seconds'] for episode in episodes]))}
        for name, episodes in (('unadapted', unadapted), ('steering in this form', steering))}
    for name, row in result['played'].items():
        print(f'       played, {name}: reward {np.mean(row["reward"]):.2f}, cost {np.mean(row["cost"]):.1f}, '
              f'{row["seconds an episode"]:.1f} s an episode', flush=True)

    (out / 'check_memory.json').write_text(json.dumps(result, indent=2) + '\n')
    if failed:
        raise SystemExit(f'FAILED: {", ".join(failed)}')
    print('every check holds')


if __name__ == '__main__':
    main()
