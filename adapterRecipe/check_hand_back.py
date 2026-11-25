"""Asks whether a short push that hands back to the agent gets a stalled robot going, and whether the world model sees it.

The look-ahead holds each action for its whole length (look_ahead.py). Held for 15
decisions a turn is a circle and not a way round a hazard, so nothing an adapter reads
shows a detour paying off. The other way, which look_ahead.py leaves to be measured, is a
short push after which the agent carries on. The frozen agent reaches the goal from
anywhere, so a few decisions to one side may be all a detour needs. Before that look-ahead
is built, this measures two things at states where a robot with steering really stalled:

    Is there a way out?      Each future is played in the simulator.
    Does the model see it?   The same futures are imagined and set beside what followed.

The stalls come from saved episodes of steering (scoreboard.py --keep). The robot is
stuck at a decision if over the next --span decisions it covers under a tenth of what the
unadapted agent usually covers in as many. One state is taken from each stall, a few
decisions after it began.

The futures tried from a stall:

    hand back        the agent alone, from the stall on
    number N up      action number N at +1 for --push-for decisions, the other numbers as
    number N down    the agent proposed them at the stall (or at -1); then the agent alone
    stopped          the zero action for --push-for decisions, then the agent alone

To play one, the saved episode is played again from its start with its saved actions put
where an adapter's would go, up to the stall. Those decisions must give the saved rewards
and costs exactly, or the check fails. What really followed is the reward, the cost and
the distance covered over the --after decisions after the stall. `as it was` is what
steering itself went on to do there, from the saved episode.

A way out is a future that really cost nothing and really earned at least half of what
the unadapted agent usually earns in as many decisions.

The same futures are imagined in three ways, each read at every length of --lengths as
the look-ahead reads: the expected cost added up, and the reward added up with the value
estimate of the last state.

    then agent, likely   the push and then the actor's most likely action, in the model's
                         most likely next states. This is what an adapter would read
    then agent, drawn    the same with random draws, as a mean over --draws rolls
    held                 the present look-ahead: each action held for the whole length,
                         and the proposed action held where the other two hand back

Measured:

    stalls     how near the nearest object of each kind is at the stalls, against all
               decisions of the same episodes
    futures    what really followed each future, as a mean over the stalls
    way out    at how many stalls some future is a way out, and what really followed the
               best future in hindsight (the most reward among those that cost nothing,
               else the least cost) and a future taken blindly
    choice     for each way of imagining: what really followed the future it would
               choose, which is the one with the most imagined reward among those imagined
               to cost under half a unit, else the one imagined to cost least
    order      how well the imagined order of the futures at a stall agrees with the real
               order (rank agreement, as a mean over the stalls)
    sides      for each action number: the imagined difference between pushing it up and
               pushing it down, against the real difference, over the stalls

From this folder:

  ../msCli/ms submit -t 4h -n check-hand-back -- bash on_task.sh check_hand_back.py --predictor PREDICTOR \\
      --episodes RUN_FOLDER/episodes --task=PointGoal1
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np

from check_stall import rank_agreement
from cost_predictor import files_of, read
from episodes import Player
from look_ahead import expected


def covered(x, y, span):
    """The distance between where the robot is at each entry and where it is `span` entries later."""
    return np.hypot(x[span:] - x[:-span], y[span:] - y[:-span])


def stalls(episode, usual, span, into, room):
    """The entries of an episode at which a stall is tried: one per stall, `into` decisions after it began.

    An entry is left out if the stall is over by then, or if fewer than `room` decisions follow it.
    """
    stuck = np.concatenate([[False], covered(episode['x'], episode['y'], span) < usual / 10, [False]])
    began = np.flatnonzero(stuck[1:] & ~stuck[:-1])
    over = np.flatnonzero(~stuck[1:] & stuck[:-1])
    last = len(episode['cost']) - 1
    return [int(start + into) for start, end in zip(began, over) if end - start > into and start + into + room <= last]


def pushes(proposed):
    """The actions tried at a stall, in the order of look_ahead.held: the proposed one, each number at +1 and at -1, and zero."""
    base = np.clip(proposed.astype(np.float32), -1, 1)
    actions = [base]
    for number in range(len(base)):
        for limit in (1.0, -1.0):
            moved = base.copy()
            moved[number] = limit
            actions.append(moved)
    actions.append(np.zeros_like(base))
    return np.stack(actions)


def names(action_numbers):
    """What each future is, in the order of pushes()."""
    return (['hand back'] + [f'number {number} {way}' for number in range(action_numbers) for way in ('up', 'down')]
            + ['stopped'])


def played(player, saved, entry, action, push_for):
    """Plays a saved episode again up to `entry`, then `action` for `push_for` decisions, then the agent alone to the end."""
    done = 0

    def adapter(state, proposed):
        nonlocal done
        at, done = done, done + 1
        if at < entry:
            return saved['action'][at][None]
        if at < entry + push_for:
            return action[None]
        return proposed

    episode, = player.play([int(saved['seed'])], adapter=adapter)
    return episode


def followed(episode, entry, after):
    """The reward, the cost and the distance covered over the `after` decisions after `entry`."""
    # Entry t + 1 of an episode holds what came of the action of entry t.
    window = slice(entry + 1, entry + 1 + after)
    distance = np.hypot(episode['x'][entry + after] - episode['x'][entry], episode['y'][entry + after] - episode['y'][entry])
    return {'reward': float(episode['reward'][window].sum()), 'cost': float(episode['cost'][window].sum()), 'distance': float(distance)}


def best(reward, cost, bar=0.0):
    """For each stall, the future with the most reward among those that cost `bar` or less, else the one that cost least."""
    within = cost <= bar
    return np.where(within.any(1), np.where(within, reward, -np.inf).argmax(1), cost.argmin(1))


def mean_rank_agreement(imagined, real):
    """The rank agreement between imagined and real over the futures of one stall, as a mean over the stalls that have one."""
    each = [rank_agreement(one, other) for one, other in zip(imagined, real)]
    each = [value for value in each if value is not None]
    return float(np.mean(each)) if each else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True, help='the cost predictor of the task, whose episodes are the unadapted agent\'s')
    parser.add_argument('--episodes', required=True, nargs='+', help='folders of steering episodes saved by scoreboard.py --keep')
    parser.add_argument('--stalls', type=int, default=80, help='how many stalls to try')
    parser.add_argument('--push-for', type=int, default=5, help='decisions a push lasts before the agent carries on')
    parser.add_argument('--after', type=int, default=40, help='decisions after the stall over which what really followed is added up')
    parser.add_argument('--lengths', type=int, nargs='+', default=[15, 40], help='the lengths at which the imagined futures are read')
    parser.add_argument('--draws', type=int, default=4, help='rolls with random draws from each stall')
    parser.add_argument('--span', type=int, default=20, help='decisions over which standing still is judged')
    parser.add_argument('--into', type=int, default=5, help='how many decisions after a stall began its state is taken')
    parser.add_argument('--seed', type=int, default=0, help='of the choice of stalls')
    args = parser.parse_args()

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    predictor, trained, _ = read(args.predictor, args.checkpoint)
    for_gpu = {name: predictor[name] for name in predictor if name != 'ahead'}
    player = Player(args.checkpoint)
    player.watch = True
    frozen = player.frozen
    lengths = sorted(set(args.lengths))
    called = names(player.action_numbers)

    # What is usual: the unadapted agent's own episodes, those the predictor began with. They were not
    # watched when they were played, so the simulator is stepped through their actions again for the places.
    alone = [dict(np.load(file)) for part in ('validation', 'training') for file in files_of(trained, part)]
    usual_reward = sum(float(episode['reward'].sum()) for episode in alone) / sum(len(episode['cost']) - 1 for episode in alone)
    distances = []
    for episode in alone:
        _, costs, seen = player.replay_watched(int(episode['seed']), episode['action'][:-1])
        if not np.array_equal(costs, episode['cost'][1:]):
            raise SystemExit(f'seed {int(episode["seed"])}: stepping the simulator through the saved actions again did not give the saved costs')
        distances.append(covered(seen['x'], seen['y'], args.span))
    usual = float(np.median(np.concatenate(distances)))
    bar = 0.5 * usual_reward * args.after

    # The stalls, from the saved episodes of steering.
    files = sorted(file for folder in args.episodes for file in pathlib.Path(folder).glob('seed_*.npz'))
    saved = [dict(np.load(file)) for file in files]
    found = [(number, entry) for number, episode in enumerate(saved)
             for entry in stalls(episode, usual, args.span, args.into, max(args.after, args.push_for))]
    stuck_share = float(np.mean(np.concatenate([covered(episode['x'], episode['y'], args.span) < usual / 10 for episode in saved])))
    if not found:
        raise SystemExit('no stall in these episodes')
    random = np.random.default_rng(args.seed)
    chosen = sorted(found[pick] for pick in random.choice(len(found), min(args.stalls, len(found)), replace=False))
    print(f'{player.task}: the unadapted agent covers {usual:.3f} in {args.span} decisions and earns {usual_reward * args.after:.2f} in {args.after}. '
          f'{len(saved)} steering episodes, stuck at {stuck_share:.1%} of decisions, {len(found)} stalls, {len(chosen)} tried', flush=True)

    near = sorted(name for name in saved[0] if name.startswith('near_'))
    where = {name[len('near_'):]: {
        'at the stalls': float(np.median([saved[number][name][entry] for number, entry in chosen])),
        'at all decisions': float(np.median(np.concatenate([episode[name] for episode in saved])))} for name in near}
    print('nearest object, median distance at the stalls (at all decisions): '
          + ', '.join(f'{kind} {both["at the stalls"]:.2f} ({both["at all decisions"]:.2f})' for kind, both in where.items()), flush=True)

    # Each future of each stall, played in the simulator.
    started = time.perf_counter()
    actions = np.stack([pushes(saved[number]['proposed'][entry]) for number, entry in chosen])
    stalled, futures = actions.shape[:2]
    real = {name: np.zeros((stalled, futures)) for name in ('reward', 'cost', 'distance')}
    was = {name: np.zeros(stalled) for name in real}
    unmatched, state_gap = 0, 0.0
    for row, (number, entry) in enumerate(chosen):
        episode = saved[number]
        for name, value in followed(episode, entry, args.after).items():
            was[name][row] = value
        for future in range(futures):
            again = played(player, episode, entry, actions[row, future], 0 if future == 0 else args.push_for)
            if (not np.array_equal(again['cost'][:entry + 1], episode['cost'][:entry + 1])
                    or np.abs(again['reward'][:entry + 1] - episode['reward'][:entry + 1]).max() > 1e-5):
                unmatched += 1
                continue
            state_gap = max(state_gap, float(np.abs(again['deter'][entry].astype(np.float32) - episode['deter'][entry]).max()))
            for name, value in followed(again, entry, args.after).items():
                real[name][row, future] = value
        if (row + 1) % 10 == 0:
            print(f'{row + 1} of {stalled} stalls played, {time.perf_counter() - started:.0f} s', flush=True)

    # The same futures, imagined. One imagined decision is compiled and a roll is a plain loop over it, as in imagine.py.
    def decision_with(sample):
        def decide(state, given, forced, predictor):
            xp = frozen.xp
            state = {name: state[name] for name in frozen.names}
            proposed = frozen.as_float(frozen.propose(state, sample=sample))
            action = xp.where(forced[:, None], given, proposed)
            following, reward = frozen.step(state, action, sample=sample)
            return {'state': following, 'cost': expected(xp, predictor, frozen.numbers(following)),
                    'reward': reward, 'value': frozen.worth(following)}
        return frozen.compiled(decide)

    # One row per future of every stall: the stalls in order, and for each its futures in order.
    kept = {name: np.concatenate([np.repeat(frozen.kept(saved[number], slice(entry, entry + 1))[name], futures, 0)
                                  for number, entry in chosen]) for name in frozen.names}
    start = frozen.written_out(kept)
    given = actions.reshape(stalled * futures, -1)
    hand_back = np.tile([0] + [args.push_for] * (futures - 1), stalled)
    never = np.full(stalled * futures, lengths[-1])

    def imagined(decide, pushed_until, seed=0):
        """The cost and the reward of every future at each of `lengths`, with the action forced for `pushed_until` decisions."""
        frozen.reseed(seed)
        state, cost, reward, read_at = start, 0.0, 0.0, {}
        for decision in range(1, lengths[-1] + 1):
            came = decide(state, given, decision <= pushed_until, for_gpu)
            state, cost, reward = came['state'], cost + came['cost'], reward + came['reward']
            if decision in lengths:
                # The value estimate stands for the reward still to come after the last imagined decision.
                read_at[decision] = {'cost': cost.reshape(stalled, futures), 'reward': (reward + came['value']).reshape(stalled, futures)}
        return read_at

    likely, drawn = decision_with(False), decision_with(True)
    rolls = [imagined(drawn, hand_back, seed) for seed in range(args.draws)]
    ways = {
        'then agent, likely': imagined(likely, hand_back),
        'then agent, drawn': {length: {name: np.mean([roll[length][name] for roll in rolls], 0) for name in ('cost', 'reward')}
                              for length in lengths},
        'held': imagined(likely, never),
    }

    # What it shows.
    out_there = (real['cost'] == 0) & (real['reward'] >= bar)
    at = np.arange(stalled)

    def taken(choice):
        """What really followed the future chosen at each stall, as means over the stalls."""
        return {'real reward': float(real['reward'][at, choice].mean()), 'real cost': float(real['cost'][at, choice].mean()),
                'really cost nothing': float((real['cost'][at, choice] == 0).mean()), 'a way out': float(out_there[at, choice].mean())}

    hindsight = best(real['reward'], real['cost'])
    way_out = {
        'bar: half the usual reward over the decisions after': bar,
        'stalls with a way out': int(out_there.any(1).sum()), 'stalls': stalled,
        'as it was': {'real reward': float(was['reward'].mean()), 'real cost': float(was['cost'].mean()),
                      'really cost nothing': float((was['cost'] == 0).mean()),
                      'a way out': float(((was['cost'] == 0) & (was['reward'] >= bar)).mean())},
        'the best future in hindsight': taken(hindsight),
        'a future taken blindly': {'real reward': float(real['reward'].mean()), 'real cost': float(real['cost'].mean()),
                                   'really cost nothing': float((real['cost'] == 0).mean()), 'a way out': float(out_there.mean())},
    }
    by_future = {name: {'real reward': float(real['reward'][:, future].mean()), 'real cost': float(real['cost'][:, future].mean()),
                        'distance': float(real['distance'][:, future].mean()), 'really cost nothing': float((real['cost'][:, future] == 0).mean()),
                        'a way out': float(out_there[:, future].mean())} for future, name in enumerate(called)}
    by_future['as it was'] = {**way_out['as it was'], 'distance': float(was['distance'].mean())}

    shows = {}
    for way, read_at in ways.items():
        for length, numbers in read_at.items():
            sides = {}
            for number in range(player.action_numbers):
                up, down = 1 + 2 * number, 2 + 2 * number
                sides[f'number {number}'] = {}
                for name in ('reward', 'cost'):
                    really = real[name][:, up] - real[name][:, down]
                    thought = numbers[name][:, up] - numbers[name][:, down]
                    differ = really != 0
                    sides[f'number {number}'][name] = {
                        'rank agreement': rank_agreement(thought, really), 'stalls where the sides really differ': int(differ.sum()),
                        'of those, the better side is imagined better': float((np.sign(thought[differ]) == np.sign(really[differ])).mean())
                        if differ.any() else None}
            shows[f'{way}, read at {length}'] = {
                'choice': taken(best(numbers['reward'], numbers['cost'], 0.5)),
                'order': {name: mean_rank_agreement(numbers[name], real[name]) for name in ('reward', 'cost')},
                'over all futures': {name: rank_agreement(numbers[name].ravel(), real[name].ravel()) for name in ('reward', 'cost')},
                'sides': sides}

    def line(name, row):
        return (f'  {name:34s} reward {row["real reward"]:6.2f}  cost {row["real cost"]:6.2f}  '
                f'cost nothing {row["really cost nothing"]:4.0%}  a way out {row["a way out"]:4.0%}')

    print(f'\nWhat really followed over {args.after} decisions, a push lasting {args.push_for}. Replays that did not match: {unmatched}; '
          f'largest gap between the state at a stall and the saved one {state_gap:.2g}')
    for name, row in by_future.items():
        print(line(name, row) + f'  distance {row["distance"]:.2f}')
    print(f'A way out (no cost, reward of {bar:.2f} or more) at {way_out["stalls with a way out"]} of {stalled} stalls')
    for name in ('the best future in hindsight', 'a future taken blindly'):
        print(line(name, way_out[name]))
    print('The future each way of imagining would choose:')
    for name, show in shows.items():
        print(line(name, show['choice']))
    print('Rank agreement of imagined with real (None: nothing to order). Order within a stall, then over all futures, then between the two sides:')
    for name, show in shows.items():
        def shown(value):
            return ' none' if value is None else f'{value:+.2f}'
        sides = '; '.join(f'{which} reward {shown(both["reward"]["rank agreement"])} cost {shown(both["cost"]["rank agreement"])}'
                          for which, both in show['sides'].items())
        print(f'  {name:34s} reward {shown(show["order"]["reward"])} cost {shown(show["order"]["cost"])} | '
              f'reward {shown(show["over all futures"]["reward"])} cost {shown(show["over all futures"]["cost"])} | {sides}', flush=True)

    (out / 'check_hand_back.json').write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name, 'task': player.task, 'predictor': args.predictor,
        'episodes': [str(folder) for folder in args.episodes], 'settings': vars(args),
        'usual': {'distance in a span': usual, 'reward a decision': usual_reward},
        'stuck share of decisions': stuck_share, 'stalls found': len(found),
        'replays that did not match': unmatched, 'largest state gap at a stall': state_gap,
        'nearest object, median distance': where, 'futures': by_future, 'way out': way_out, 'imagined': shows,
        # Everything by stall and future, for looking closer without playing again.
        'each stall': {'episode file': [str(files[number]) for number, _ in chosen], 'entry': [entry for _, entry in chosen],
                       'future names': called, 'real': {name: values.tolist() for name, values in real.items()},
                       'as it was': {name: values.tolist() for name, values in was.items()},
                       'imagined': {way: {str(length): {name: np.asarray(values).tolist() for name, values in numbers.items()}
                                          for length, numbers in read_at.items()} for way, read_at in ways.items()}},
    }, indent=2) + '\n')
    if unmatched:
        raise SystemExit('FAIL: some replays did not give what the episode recorded up to the stall')
    print('\nPASS: every replay matched its episode up to the stall; the rest above is measured, not judged')


if __name__ == '__main__':
    main()
