"""Searches for an adapter: many candidates scored in the world model, a few real episodes to keep it honest.

This is the recipe of brainStorming/adapterRecipeDesign.md and of
implementation/adapterArchitecture.md put together. It starts from an adapter that does
nothing and ends with one saved adapter. With --begin steering it starts from steering
written by hand in the adapter's form, to see what a search adds to that.

One generation, with no simulator:

    starting states are drawn, the same for every candidate of the generation
    the search (cma.py) gives its candidates
    each is rolled in the world model from those states, with the same random numbers
    a candidate is scored, and the search is told the scores

A roll gives a candidate's reward and cost as shares of the unadapted agent's, rolled from
the same states with the same random numbers: 1 means as much as doing nothing. It also
gives the share of the proposed action's size that the candidate keeps.

Every few generations comes a round of real episodes, a check. The search's current best
plays in the simulator, and from those episodes:

    the predictor   is fitted again, to the unadapted episodes it began with and every
                    episode played so far
    the states      of those episodes become the starting states of the next generations,
                    each with the memory the adapter had there. Two kinds are kept for
                    good, and each makes up a share of every generation's states: those at
                    which a cost was charged with the few decisions before them, and the
                    whole of the episode that earned the least. The second is how a stall
                    gets among the starting states without anything having to detect one

How a candidate is scored, and which adapter is the current best, is one of two ways.

With one score (the default):

    score           reward share - price x cost share
    current best    the centre of the search
    the price       goes up if the real episodes cost more than the limit and down if
                    less, by the ratio of the two, but by no more than a factor of 1.5

With one score and --hold, moving the action is charged for as well:

    score           reward share - price x cost share - hold x change
    change          how far the action taken lies from the proposed one, as a mean over
                    the roll and the action's numbers
    the hold        goes up if the real episodes earned a smaller share of the unadapted
                    agent's reward than the rolls imagined and down if a larger one, by
                    the ratio of the two, but by no more than a factor of 1.5

With an archive (--archive, archive.py):

    score           how the candidate changed the archive, which keeps the adapter with
                    the most reward for every cost level and manner of acting. Every
                    adapter it holds is rolled again in each generation
    current best    the adapter with the most reward at or under the limit level
    the limit level is the imagined cost level that matches the real limit. It moves
                    towards the level the real episodes point to: the imagined cost level
                    of the adapter that played, times the limit over its real cost. A
                    change to the archive counts for more the nearer its cost level is to
                    it. After every check, and when a generation adds nothing to the
                    archive, the search starts again from an adapter near that level

In both ways --damped shrinks the most the price or the level changes each time the real
cost crosses the limit, and --cautious holds an adapter to its mean cost plus the
standard error. It shrinks the most the hold changes in the same way, each time the real
reward share crosses the imagined one.

At the end the adapters that played best are finalists. Each plays more episodes, on the
same seeds as the other finalists, and the one with the most reward among those within the
cost limit over all its episodes is saved. If none is within the limit, the one with the
least cost is.

Why it is built this way:

  No price of cost is set by hand. What is set is the limit, which comes with the task,
  and the price follows from the real episodes.

  A world model is most to be trusted near what its agent does. Where the real episodes
  earn less than the rolls imagined, the adapter has moved the action further than the
  model can follow, and the hold makes moving it cost more. Like the price it is set by
  the real episodes and not by hand.

  An adapter is saved together with the predictor it was played with. The predictor
  changes during the search, and the same weights with another predictor are another
  adapter.

  The real episodes are the budget. With the defaults: 15 checks of 10 episodes, 3
  finalists with 20 more each, and the 40 episodes behind the first predictor, which is
  250 episodes or 250,000 simulator steps. final.json says how many were used and what
  cost was run up in them.

Written to the run folder:

    search.jsonl       one line per generation
    checks.jsonl       one line per check: the reward and cost of every episode
    adapters/          the adapter of every check, as adapter.save writes it
    archive.jsonl      with --archive, the archive at every check
    final.npz          the chosen adapter; scoreboard.py --kind saved plays it
    final.json         how it was chosen and what the search used

From this folder:

  ../msCli/ms submit -t 6h -n search -- apptainer exec --nv --bind /datasets/smohan \\
      /home-mscluster/smohan/images/safeDreamer.sif python3 search.py --seed 0 \\
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
from archive import Archive, LimitLevel
from cma import CMA, RandomSearch
from imagine import Imagination
from look_ahead import measured_by_stretch

MOST_THE_PRICE_CHANGES = 1.5    # at one check, up or down


def player_of(checkpoint):
    """The frozen agent of `checkpoint` with its simulator: DreamerV3 on Safety-Gymnasium, or TD-MPC2 on Meta-World."""
    if str(checkpoint).endswith('.pt'):
        from frozen_tdmpc2 import Player
    else:
        from episodes import Player
    return Player(checkpoint)


def places(frozen, episode, before, memory):
    """The states of an episode a roll can start from, each with the adapter's memory there, and which are failures.

    A failure is a state at which a cost was charged, or one of the `before` decisions
    leading up to it. The last entry of an episode is left out: no action was taken there.
    An episode played without a memory starts every roll with the empty one, `memory`.
    """
    last = len(episode['cost']) - 1
    states = frozen.kept(episode, slice(0, last))
    states['memory'] = episode['memory'][:last] if 'memory' in episode else np.repeat(memory, last, 0)
    cost = episode['cost'][:last]
    failed = np.zeros(last, bool)
    for entry in np.flatnonzero(cost > 0):
        failed[max(0, entry - before):entry + 1] = True
    return states, failed


def together(parts):
    """Several sets of states as one."""
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def only(states, chosen):
    """The chosen rows of a set of states."""
    return {name: rows[chosen] for name, rows in states.items()}


def drawn(frozen, latest, kept, count, random):
    """`count` starting states with the memory at each.

    kept    the sets of states that are kept for good, each with the share of the starting
            states it makes up, as (states, share). A set that is empty, or smaller than
            its share, gives what it has. The rest come from `latest`.
    """
    picked, left = [], count
    for states, share in kept:
        have = len(states['memory']) if states else 0
        number = min(int(share * count), have)
        if number:
            picked.append(only(states, random.choice(have, number, replace=False)))
            left -= number
    picked.append(only(latest, random.choice(len(latest['memory']), left, replace=len(latest['memory']) < left)))
    picked = together(picked)
    memory = picked.pop('memory')
    return frozen.written_out(picked), memory


def judged(costs, cautious):
    """The cost an adapter is held to: the mean of its episodes, or with `cautious` that mean plus its standard error.

    Cost sits in a few episodes, so the mean of a handful is often far under the true
    one. The standard error is large exactly when some episode was costly.
    """
    costs = np.asarray(costs, np.float64)
    error = costs.std(ddof=1) / np.sqrt(len(costs)) if cautious and len(costs) > 1 else 0.0
    return float(costs.mean() + error)


def best_of(played, count, limit):
    """The `count` best of adapters that have played: the most reward among those within the cost limit, then the least cost."""
    within = sorted((one for one in played if one['judged cost'] <= limit), key=lambda one: -one['reward'])
    beyond = sorted((one for one in played if one['judged cost'] > limit), key=lambda one: one['judged cost'])
    return (within + beyond)[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True, help='the cost_predictor.npz the search begins with')
    parser.add_argument('--seed', type=int, default=0, help='of the search; it also sets which real episodes are played')
    parser.add_argument('--search', default='cma', choices=('cma', 'random'))
    parser.add_argument('--form', default='memory', choices=('memory',) + adapters.FORMS)
    parser.add_argument('--no-memory', action='store_true', help='form memory: leave the memory out')
    parser.add_argument('--added-on', action='store_true', help='form memory: give every output an added-on weight')
    parser.add_argument('--no-stop', action='store_true', help='form memory: leave the stopped action out of the look-ahead')
    parser.add_argument('--look', default='likely', choices=('likely', 'mean'),
                        help='form memory: the look-ahead rolls on the most likely classes or on the mean of the choices')
    parser.add_argument('--size', default='tens', choices=adapters.SIZES, help='of a residual or a mix')
    parser.add_argument('--decisions', type=int, default=10, help='how many decisions the look-ahead of the first forms lasts')
    parser.add_argument('--push', type=float, default=1.0)
    parser.add_argument('--begin', default='zero', choices=('zero', 'steering'),
                        help='form memory: the adapter the search starts from')
    parser.add_argument('--step', type=float, default=0.3, help='how far from that adapter the first candidates lie')
    parser.add_argument('--candidates', type=int, default=16, help='in one generation')
    parser.add_argument('--diagonal', action='store_true', help='CMA-ES with one number per weight in its table')
    parser.add_argument('--starts', type=int, default=256, help='starting states of a generation')
    parser.add_argument('--failures', type=float, default=0.25, help='the share of them that are kept costly states')
    parser.add_argument('--lowest', type=float, default=0.25,
                        help='the share of them from the kept lowest-earning episodes; 0 keeps none')
    parser.add_argument('--roll', type=int, default=40, help='decisions of an imagined roll')
    parser.add_argument('--generations', type=int, default=5, help='between two checks')
    parser.add_argument('--checks', type=int, default=15)
    parser.add_argument('--episodes', type=int, default=10, help='real episodes of a check')
    parser.add_argument('--finalists', type=int, default=3)
    parser.add_argument('--more', type=int, default=20, help='further real episodes of each finalist')
    parser.add_argument('--limit', type=float,
                        help='the cost limit: the highest mean episode cost that counts as safe (default: the task\'s own)')
    parser.add_argument('--archive', action='store_true', help='score by an archive over cost level and manner of acting')
    parser.add_argument('--keep-predictor', action='store_true', help='do not fit the cost predictor again after a check')
    parser.add_argument('--cautious', action='store_true',
                        help='hold an adapter to its mean cost plus the standard error, for the price and for the final choice')
    parser.add_argument('--damped', action='store_true',
                        help='the most the price changes at a check shrinks each time the real cost crosses the limit')
    parser.add_argument('--hold', action='store_true',
                        help='one score: also charge for moving the action, more when the real reward falls short of the imagined')
    parser.add_argument('--first-seed', type=int, help='of the real episodes (default: 100000 + 1000 x seed)')
    args = parser.parse_args()

    from cost_predictor import files_of, fit, read, windows
    from episodes import make_folder

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    make_folder(out / 'adapters')
    random = np.random.default_rng(args.seed)
    predictor, trained, _ = read(args.predictor, args.checkpoint)
    if trained['ahead'] != 0:
        raise SystemExit('the look-ahead needs a predictor of the cost of arriving in a state (--ahead 0)')
    for_gpu = {name: predictor[name] for name in predictor if name != 'ahead'}
    began_with = [file for part in ('validation', 'training') for file in files_of(trained, part)]
    training = files_of(trained, 'training')

    player = player_of(args.checkpoint)
    frozen = player.frozen
    action_numbers = player.action_numbers
    if args.limit is None:
        args.limit = player.cost_limit
    if args.form == 'memory':
        adapter = adapters.MemoryAdapter(action_numbers, memory=not args.no_memory, added_on=args.added_on, stop=not args.no_stop,
                                         roll=args.look)
        fixed = {'predictor': for_gpu,
                 'sizes': measured_by_stretch(frozen, for_gpu, training, adapter.lengths, adapter.stop, adapter.roll)}
        before, empty = max(adapter.lengths), adapter.start()
        act = adapter.compiled(frozen)
    else:
        from scoreboard import measured_sizes
        state_numbers = int(for_gpu['weights'].shape[0])
        adapter = adapters.Adapter(args.form, None if args.form == 're-rank' else args.size,
                                   action_numbers, state_numbers, args.decisions, args.push)
        fixed = {'predictor': for_gpu, 'sizes': measured_sizes(frozen, for_gpu, training, args.decisions, args.push)}
        before, empty = args.decisions, np.zeros((1, 1), np.float32)
        act = adapter.compiled(frozen)
    remembers = args.form == 'memory'

    def rows_for_predictor(episode):
        """An episode as the rows the cost predictor is fitted to: the state numbers and the cost of arriving."""
        entries, amount, _ = windows(episode['cost'], 0)
        return frozen.rows(frozen.kept(episode, entries)), amount

    # The starting states begin as those of the unadapted episodes behind the predictor.
    unadapted = [np.load(file) for file in began_with]
    seen = [places(frozen, episode, before, empty) for episode in unadapted]
    latest = together([states for states, _ in seen])
    failures = together([only(states, failed) for states, failed in seen])
    lowest = None
    cost_behind_predictor = float(sum(episode['cost'].sum() for episode in unadapted))
    unadapted_cost = cost_behind_predictor / len(unadapted)
    unadapted_reward = float(np.mean([episode['reward'].sum() for episode in unadapted]))
    decisions_an_episode = len(unadapted[0]['cost']) - 1

    # The rows the predictor is fitted to: room for every episode the search will play.
    first_rows = [rows_for_predictor(np.load(file)) for file in training]
    most_rows = sum(len(amount) for _, amount in first_rows) + (
        args.checks * args.episodes + args.finalists * args.more) * decisions_an_episode
    numbers = np.zeros((most_rows, first_rows[0][0].shape[1]), np.float32)
    amounts = np.zeros(most_rows, np.float32)
    filled = 0
    for new_numbers, new_amounts in first_rows:
        numbers[filled:filled + len(new_amounts)], amounts[filled:filled + len(new_amounts)] = new_numbers, new_amounts
        filled += len(new_amounts)
    del first_rows, unadapted

    zero = np.zeros(adapter.count, np.float32)
    if args.begin == 'steering' and not remembers:
        raise SystemExit('--begin steering is for the adapter with memory (--form memory)')
    begin = adapter.steering() if args.begin == 'steering' else zero

    def new_search(centre, seed):
        if args.search == 'random':
            return RandomSearch(centre, args.step, args.candidates, seed)
        return CMA(centre, args.step, args.candidates, seed, diagonal=args.diagonal)

    search = new_search(begin, args.seed)
    restarts = 0
    kept = [(failures, args.failures), (lowest, args.lowest)]
    starts, memory = drawn(frozen, latest, kept, args.starts, random)
    imagination = Imagination(frozen, adapter, fixed, starts)
    seeds = iter(range(100000 + 1000 * args.seed if args.first_seed is None else args.first_seed, 10 ** 9))
    price, generation, checked, cost_paid = 1.0, 0, [], 0.0
    most, was_beyond = MOST_THE_PRICE_CHANGES, None
    if args.hold and args.archive:
        raise SystemExit('--hold is for the search with one score')
    hold, most_hold, was_short = (1.0 if args.hold else 0.0), MOST_THE_PRICE_CHANGES, None
    archive = Archive() if args.archive else None
    if archive is not None:
        # The adapter that does nothing is the first the archive holds: as much reward and cost as the unadapted agent.
        archive.rebuild([{'weights': zero, 'reward': 1.0, 'cost': 1.0, 'kept': 1.0}])
        if args.begin == 'steering':
            # And the adapter the search starts from. Its cell here is a guess: it is scored in the first generation.
            archive.rebuild(list(archive.members.values()) + [{'weights': begin, 'reward': 0.5, 'cost': 0.25, 'kept': 0.6}])
    limit_level = LimitLevel(args.limit, unadapted_cost, MOST_THE_PRICE_CHANGES, args.damped)
    print(f'{player.task}, search seed {args.seed}, {args.search}{" with an archive" if args.archive else ""}: '
          f'{adapter.describe()}, {args.candidates} candidates a generation, {args.starts} starting states, '
          f'rolls of {args.roll} decisions; cost limit {args.limit:g}, unadapted cost {unadapted_cost:.1f}', flush=True)

    def measured(weights, reference, seed):
        """What a roll with `weights` shows, against the unadapted roll `reference` of the same seed and states."""
        roll = imagination.roll(np.asarray(weights, np.float32), args.roll, seed, starts, fixed, memory)
        # A share of 1 is as much as the unadapted agent. Written as a difference so that it also holds where rewards are below zero.
        earned, usual = roll['reward'].mean(), reference['reward'].mean()
        return {'reward': float(1 + (earned - usual) / max(abs(usual), 1e-9)),
                'cost': float(roll['cost'].mean() / max(reference['cost'].mean(), 1e-9)),
                'kept': float(roll['size taken'].sum() / max(roll['size proposed'].sum(), 1e-9)),
                'moved': float(roll['moved'].mean()),
                'change': float(roll['change'].mean())}

    def played(weights, count):
        """Plays `count` real episodes with the adapter at `weights`; returns their rewards and costs and takes in what they showed."""
        nonlocal filled, cost_paid
        weights = np.asarray(weights, np.float32)
        if remembers:
            episodes = player.play([next(seeds) for _ in range(count)], memory=empty,
                                   adapter=lambda state, proposed, held: act(state, proposed, held, fixed, weights))
        else:
            episodes = player.play([next(seeds) for _ in range(count)],
                                   adapter=lambda state, proposed: act(state, proposed, fixed, weights))
        rewards, costs, seen = [], [], []
        for episode in episodes:
            rewards.append(float(episode['reward'].sum()))
            costs.append(float(episode['cost'].sum()))
            seen.append(places(frozen, episode, before, empty))
            new_numbers, new_amounts = rows_for_predictor(episode)
            numbers[filled:filled + len(new_amounts)], amounts[filled:filled + len(new_amounts)] = new_numbers, new_amounts
            filled += len(new_amounts)
        cost_paid += sum(costs)
        return rewards, costs, seen

    for check in range(args.checks):
        for _ in range(args.generations):
            started = time.perf_counter()
            starts, memory = drawn(frozen, latest, kept, args.starts, random)
            reference = imagination.roll(zero, args.roll, generation, starts, fixed, memory)
            asked = search.ask()
            scored = [{'weights': np.asarray(weights, np.float32), **measured(weights, reference, generation)} for weights in asked]
            if archive is None:
                scores = [one['reward'] - price * one['cost'] - hold * one['change'] for one in scored]
                added = None
            else:
                # Every adapter the archive holds is scored again on this generation's states.
                archive.rebuild([{'weights': member['weights'], **measured(member['weights'], reference, generation)}
                                 for member in archive.members.values()])
                scores, added = archive.take_in(scored, limit_level.level)
            search.tell(scores)
            top = int(np.argmax(scores))
            line = {'generation': generation, 'price': price, 'hold': hold, 'step': float(search.step),
                    'best score': scores[top], 'best reward share': scored[top]['reward'], 'best cost share': scored[top]['cost'],
                    'mean reward share': float(np.mean([one['reward'] for one in scored])),
                    'mean cost share': float(np.mean([one['cost'] for one in scored])),
                    'mean kept share': float(np.mean([one['kept'] for one in scored])),
                    'mean change': float(np.mean([one['change'] for one in scored])),
                    'seconds': round(time.perf_counter() - started, 1)}
            if archive is not None:
                line.update({'limit level': limit_level.level, 'in the archive': len(archive), 'added': added, 'restarts': restarts})
                if not added:
                    # The search has run dry where it is. It starts again from an adapter the archive holds.
                    close = archive.near(limit_level.level)
                    restarts += 1
                    search = new_search(close[random.integers(len(close))]['weights'], args.seed + 1000 * restarts)
            with open(out / 'search.jsonl', 'a') as file:
                file.write(json.dumps(line) + '\n')
            generation += 1

        # The check: the current best plays real episodes.
        started = time.perf_counter()
        if archive is None:
            weights = search.centre.astype(np.float32)
        else:
            weights = archive.best(limit_level.level)['weights']
        imagined = measured(weights, reference, generation - 1)    # on the states and random numbers of the last generation
        file_name = f'adapters/check_{check:02d}.npz'
        adapters.save(out / file_name, adapter, weights, fixed)
        rewards, costs, seen = played(weights, args.episodes)
        record = {'check': check, 'generation': generation, 'adapter': file_name, 'price': price, 'hold': hold,
                  'limit level': limit_level.level,
                  'reward': float(np.mean(rewards)), 'cost': float(np.mean(costs)), 'judged cost': judged(costs, args.cautious),
                  'imagined reward share': imagined['reward'], 'imagined cost share': imagined['cost'],
                  'imagined kept share': imagined['kept'], 'imagined change': imagined['change'],
                  'real reward share': float(np.mean(rewards)) / unadapted_reward if unadapted_reward > 0 else None,
                  'rewards': rewards, 'costs': costs}
        checked.append(record)

        beyond = record['judged cost'] > args.limit
        if args.damped and was_beyond is not None and beyond != was_beyond:
            most = float(np.sqrt(most))
        was_beyond = beyond
        price *= float(np.clip(record['judged cost'] / args.limit, 1 / most, most))
        if args.hold and record['real reward share'] is not None:
            # The share of the unadapted agent's reward that was really earned, against the share the rolls imagined.
            real_share = record['real reward share']
            short = real_share < imagined['reward']
            if args.damped and was_short is not None and short != was_short:
                most_hold = float(np.sqrt(most_hold))
            was_short = short
            hold *= float(np.clip(imagined['reward'] / real_share if real_share > 0 else most_hold, 1 / most_hold, most_hold))
        # With an archive the adapter that played is often not at the limit level, so its imagined cost level counts.
        limit_level.moved_by(record['judged cost'], imagined['cost'] if archive is not None else None)
        latest = together([states for states, _ in seen])
        failures = together([failures] + [only(states, failed) for states, failed in seen])
        if args.lowest:
            # The whole of the episode that earned the least, which is where a stall is if there was one.
            least = seen[int(np.argmin(rewards))][0]
            lowest = least if lowest is None else together([lowest, least])
        kept = [(failures, args.failures), (lowest, args.lowest)]
        if not args.keep_predictor:
            costly = amounts[:filled] > 0
            start = {name: value for name, value in fixed['predictor'].items() if name != 'mean_cost'}
            fitted, ended = fit(numbers[:filled], costly, trained['chosen strength'], start=start, steps=100)
            fitted['mean_cost'] = np.float32(amounts[:filled][costly].mean())
            fixed = {'predictor': fitted, 'sizes': fixed['sizes']}
            record['predictor'] = {'rows': int(filled), 'costly share': float(costly.mean()), **ended}
        record['seconds'] = round(time.perf_counter() - started, 1)
        with open(out / 'checks.jsonl', 'a') as file:
            file.write(json.dumps(record) + '\n')
        if archive is not None:
            with open(out / 'archive.jsonl', 'a') as file:
                file.write(json.dumps({'check': check, 'limit level': limit_level.level, 'cells': archive.table()}) + '\n')
            # The search starts again near the limit level, which these episodes have just moved.
            close = archive.near(limit_level.level)
            restarts += 1
            search = new_search(close[random.integers(len(close))]['weights'], args.seed + 1000 * restarts)
        (out / 'status.json').write_text(json.dumps({
            'check': check + 1, 'of': args.checks, 'reward': round(record['reward'], 2), 'cost': round(record['cost'], 2)}) + '\n')
        print(f'check {check}, generation {generation}: reward {record["reward"]:.2f}, cost {record["cost"]:.1f} '
              f'(imagined shares: reward {imagined["reward"]:.2f}, cost {imagined["cost"]:.2f}, kept {imagined["kept"]:.2f}); '
              + (f'limit level now {limit_level.level:.3f}, {len(archive)} in the archive, {restarts} restarts'
                 if archive is not None else f'price now {price:.2f}' + (f', hold now {hold:.2f}' if args.hold else ''))
              + f', {len(failures["memory"])} costly states kept', flush=True)

    # The final choice: the best checked adapters play more, each on the same seeds, and all their episodes count.
    finalists = best_of(checked, args.finalists, args.limit)
    further = [next(seeds) for _ in range(args.more)]
    pooled = []
    for finalist in finalists:
        _, weights, fixed = adapters.load(out / finalist['adapter'], action_numbers, None if remembers else state_numbers)
        seeds = iter(further)
        rewards, costs, _ = played(weights, args.more)
        rewards, costs = finalist['rewards'] + rewards, finalist['costs'] + costs
        pooled.append({'check': finalist['check'], 'adapter': finalist['adapter'], 'episodes': len(costs),
                       'reward': float(np.mean(rewards)), 'cost': float(np.mean(costs)), 'judged cost': judged(costs, args.cautious),
                       'reward at its check': finalist['reward'], 'cost at its check': finalist['cost']})
        print(f'finalist of check {finalist["check"]}: reward {pooled[-1]["reward"]:.2f}, cost {pooled[-1]["cost"]:.2f} '
              f'over {len(costs)} episodes', flush=True)
    chosen, = best_of(pooled, 1, args.limit)
    _, weights, with_it = adapters.load(out / chosen['adapter'], action_numbers, None if remembers else state_numbers)
    adapters.save(out / 'final.npz', adapter, weights, with_it)

    episodes_played = args.checks * args.episodes + len(finalists) * args.more
    steps_an_episode = decisions_an_episode * player.steps_per_decision
    (out / 'final.json').write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'task': player.task,
        'settings': vars(args),
        'adapter': adapter.describe(),
        'chosen': chosen,
        'within the cost limit': chosen['judged cost'] <= args.limit,
        'finalists': pooled,
        'generations': generation,
        'price at the end': price,
        'hold at the end': hold,
        'limit level at the end': limit_level.level,
        'real episodes': {'behind the first predictor': len(began_with), 'played by the search': episodes_played},
        'simulator steps': (len(began_with) + episodes_played) * steps_an_episode,
        'cost run up': {'behind the first predictor': cost_behind_predictor, 'by the search': cost_paid},
    }, indent=2) + '\n')
    print(f'chosen: the adapter of check {chosen["check"]}, reward {chosen["reward"]:.2f}, cost {chosen["cost"]:.2f} over '
          f'{chosen["episodes"]} episodes; {(len(began_with) + episodes_played) * steps_an_episode:,} simulator steps used')


if __name__ == '__main__':
    main()
