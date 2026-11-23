"""Searches for an adapter: many candidates scored in the world model, a few real episodes to keep it honest.

This is the recipe of brainStorming/adapterRecipeDesign.md put together (decisions 6, 7,
9 and 10). It starts from an adapter that does nothing and ends with one saved adapter.

One generation, with no simulator:

    starting states are drawn, the same for every candidate of the generation
    the search (cma.py) gives its candidates
    each is rolled in the world model from those states, with the same random numbers
    a candidate scores  reward share - price x cost share
    the search is told the scores

The shares are against the unadapted agent, rolled from the same states with the same
random numbers: 1 means as much reward, or as much cost, as doing nothing.

Every few generations comes a check, with the simulator. The search's current best guess
plays real episodes, and three things follow from them:

    the price       goes up if those episodes cost more than the limit and down if less,
                    by the ratio of the two, but by no more than a factor of 1.5
    the predictor   is fitted again, to the unadapted episodes it began with and every
                    episode played so far
    the states      of those episodes become the starting states of the next generations.
                    Those at which a cost was charged, and the few decisions before them,
                    are kept for good and make up a share of every generation's states

At the end the checked adapters that did best are finalists. Each plays more episodes, on
the same seeds as the other finalists, and the one with the most reward among those within
the cost limit over all its episodes is saved. If none is within the limit, the one with
the least cost is.

Why it is built this way:

  No price of cost is set by hand. What is set is the limit, which comes with the task,
  and the price follows from the real episodes.

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
    final.npz          the chosen adapter; scoreboard.py --kind saved plays it
    final.json         how it was chosen and what the search used

From this folder:

  ../msCli/ms submit -t 4h -n search -- apptainer exec --nv --bind /datasets/smohan \\
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
from cma import CMA, RandomSearch
from cost_predictor import files_of, fit, load, read, windows
from episodes import Player, make_folder, states
from imagine import Imagination
from scoreboard import COST_LIMIT, measured_sizes

MOST_THE_PRICE_CHANGES = 1.5    # at one check, up or down


def rows_for_predictor(episode):
    """An episode as the rows the cost predictor is fitted to: the state numbers and the cost of arriving."""
    entries, amount, _ = windows(episode['cost'], 0)
    state = states(episode, entries)
    numbers = np.concatenate([state['deter'], state['stoch'].reshape(len(entries), -1)], -1)
    return numbers.astype(np.float32), amount


def places(episode, before):
    """The states of an episode a roll can start from, and which of them count as failures.

    A failure is a state at which a cost was charged, or one of the `before` decisions
    leading up to it. The last entry of an episode is left out: no action was taken there.
    """
    cost = episode['cost'][:-1]
    failed = np.zeros(len(cost), bool)
    for entry in np.flatnonzero(cost > 0):
        failed[max(0, entry - before):entry + 1] = True
    return {'deter': episode['deter'][:-1], 'stoch': episode['stoch'][:-1]}, failed


def together(parts):
    """Several sets of states as one."""
    return {name: np.concatenate([part[name] for part in parts]) for name in ('deter', 'stoch')}


def drawn(latest, failures, count, share, classes, random):
    """`count` starting states: a `share` of them from the failures, if there are that many, and the rest from the latest episodes."""
    from_failures = min(int(share * count), len(failures['deter']))
    picked = []
    for source, number in ((failures, from_failures), (latest, count - from_failures)):
        rows = random.choice(len(source['deter']), number, replace=False)
        picked.append({name: source[name][rows] for name in ('deter', 'stoch')})
    picked = together(picked)
    # The pieces of frozen_agent.py want each categorical choice written out as zeros and a one.
    return {'deter': picked['deter'], 'stoch': np.eye(classes, dtype=np.float16)[picked['stoch']]}


def best_of(played, count):
    """The `count` best of adapters that have played: the most reward among those within the cost limit, then the least cost."""
    within = sorted((one for one in played if one['cost'] <= COST_LIMIT), key=lambda one: -one['reward'])
    beyond = sorted((one for one in played if one['cost'] > COST_LIMIT), key=lambda one: one['cost'])
    return (within + beyond)[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True, help='the cost_predictor.npz the search begins with')
    parser.add_argument('--seed', type=int, default=0, help='of the search; it also sets which real episodes are played')
    parser.add_argument('--search', default='cma', choices=('cma', 'random'))
    parser.add_argument('--form', default='residual', choices=adapters.FORMS)
    parser.add_argument('--size', default='tens', choices=adapters.SIZES)
    parser.add_argument('--decisions', type=int, default=10, help='how many decisions the look-ahead lasts')
    parser.add_argument('--push', type=float, default=1.0)
    parser.add_argument('--step', type=float, default=0.3, help='how far from zero the first candidates lie')
    parser.add_argument('--candidates', type=int, default=16, help='in one generation')
    parser.add_argument('--starts', type=int, default=256, help='starting states of a generation')
    parser.add_argument('--failures', type=float, default=0.25, help='the share of them that are kept failures')
    parser.add_argument('--roll', type=int, default=40, help='decisions of an imagined roll')
    parser.add_argument('--generations', type=int, default=5, help='between two checks')
    parser.add_argument('--checks', type=int, default=15)
    parser.add_argument('--episodes', type=int, default=10, help='real episodes of a check')
    parser.add_argument('--finalists', type=int, default=3)
    parser.add_argument('--more', type=int, default=20, help='further real episodes of each finalist')
    parser.add_argument('--keep-predictor', action='store_true', help='do not fit the cost predictor again after a check')
    parser.add_argument('--first-seed', type=int, help='of the real episodes (default: 100000 + 1000 x seed)')
    args = parser.parse_args()

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.'))
    make_folder(out / 'adapters')
    random = np.random.default_rng(args.seed)
    predictor, trained, _ = read(args.predictor, args.checkpoint)
    if trained['ahead'] != 0:
        raise SystemExit('the look-ahead needs a predictor of the cost of arriving in a state (--ahead 0)')
    for_gpu = {name: predictor[name] for name in ('weights', 'offset', 'mean_cost')}
    began_with = [file for part in ('validation', 'training') for file in files_of(trained, part)]
    training = files_of(trained, 'training')

    player = Player(args.checkpoint)
    frozen = player.frozen
    first = np.load(training[0])
    classes = int(first['classes'])
    action_numbers = first['proposed'].shape[1]
    state_numbers = first['deter'].shape[1] + first['stoch'].shape[1] * classes
    adapter = adapters.Adapter(args.form, None if args.form == 're-rank' else args.size,
                               action_numbers, state_numbers, args.decisions, args.push)
    fixed = {'predictor': for_gpu, 'sizes': measured_sizes(frozen, for_gpu, training, args.decisions, args.push)}
    act = adapter.compiled(frozen)

    # The starting states begin as those of the unadapted episodes behind the predictor.
    unadapted = [places(np.load(file), args.decisions) for file in began_with]
    latest = together([state for state, _ in unadapted])
    failures = together([{name: state[name][failed] for name in state} for state, failed in unadapted])
    cost_behind_predictor = float(sum(np.load(file)['cost'].sum() for file in began_with))

    # The rows the predictor is fitted to: room for every episode the search will play.
    rows = load(training, 0)
    most = len(rows['numbers']) + (args.checks * args.episodes + args.finalists * args.more) * (len(first['cost']) - 1)
    numbers, amounts = np.zeros((most, state_numbers), np.float32), np.zeros(most, np.float32)
    filled = len(rows['numbers'])
    numbers[:filled], amounts[:filled] = rows['numbers'], rows['amount']
    del rows

    zero = np.zeros(adapter.count, np.float32)
    search = (CMA if args.search == 'cma' else RandomSearch)(zero, args.step, args.candidates, args.seed)
    starts = drawn(latest, failures, args.starts, args.failures, classes, random)
    imagination = Imagination(frozen, adapter, fixed, starts)
    seeds = iter(range(100000 + 1000 * args.seed if args.first_seed is None else args.first_seed, 10 ** 9))
    price, generation, checked, cost_paid = 1.0, 0, [], 0.0
    print(f'{player.task}, search seed {args.seed}, {args.search}: {adapter.describe()}, {args.candidates} candidates a '
          f'generation, {args.starts} starting states, rolls of {args.roll} decisions', flush=True)

    def shares(weights, reference, seed):
        """The reward and the cost of a roll with `weights`, as shares of the unadapted roll `reference` of the same seed."""
        roll = imagination.roll(weights.astype(np.float32), args.roll, seed, starts, fixed)
        return float(roll['reward'].mean() / reference['reward'].mean()), float(roll['cost'].mean() / reference['cost'].mean())

    def played(weights, count):
        """Plays `count` real episodes with the adapter at `weights`; returns their rewards and costs and takes in what they showed."""
        nonlocal filled, cost_paid
        weights = weights.astype(np.float32)
        rewards, costs, seen = [], [], []
        for episode in player.play([next(seeds) for _ in range(count)],
                                   adapter=lambda state, proposed: act(state, proposed, fixed, weights)):
            rewards.append(float(episode['reward'].sum()))
            costs.append(float(episode['cost'].sum()))
            seen.append(places(episode, args.decisions))
            new_numbers, new_amounts = rows_for_predictor(episode)
            numbers[filled:filled + len(new_numbers)], amounts[filled:filled + len(new_numbers)] = new_numbers, new_amounts
            filled += len(new_numbers)
        cost_paid += sum(costs)
        return rewards, costs, seen

    for check in range(args.checks):
        for _ in range(args.generations):
            started = time.perf_counter()
            starts = drawn(latest, failures, args.starts, args.failures, classes, random)
            reference = imagination.roll(zero, args.roll, generation, starts, fixed)
            asked = search.ask()
            scored = [shares(weights, reference, generation) for weights in asked]
            scores = [reward - price * cost for reward, cost in scored]
            search.tell(scores)
            top = int(np.argmax(scores))
            line = {'generation': generation, 'price': price, 'step': float(search.step),
                    'best score': scores[top], 'best reward share': scored[top][0], 'best cost share': scored[top][1],
                    'mean reward share': float(np.mean([reward for reward, _ in scored])),
                    'mean cost share': float(np.mean([cost for _, cost in scored])),
                    'seconds': round(time.perf_counter() - started, 1)}
            with open(out / 'search.jsonl', 'a') as file:
                file.write(json.dumps(line) + '\n')
            generation += 1

        # The check: the search's best guess plays real episodes.
        started = time.perf_counter()
        weights = search.centre.astype(np.float32)
        imagined = shares(weights, reference, generation - 1)    # on the states and random numbers of the last generation
        file_name = f'adapters/check_{check:02d}.npz'
        adapters.save(out / file_name, adapter, weights, fixed)
        rewards, costs, seen = played(weights, args.episodes)
        record = {'check': check, 'generation': generation, 'adapter': file_name, 'price': price,
                  'reward': float(np.mean(rewards)), 'cost': float(np.mean(costs)),
                  'imagined reward share': imagined[0], 'imagined cost share': imagined[1],
                  'rewards': rewards, 'costs': costs}
        checked.append(record)

        price *= float(np.clip(record['cost'] / COST_LIMIT, 1 / MOST_THE_PRICE_CHANGES, MOST_THE_PRICE_CHANGES))
        latest = together([state for state, _ in seen])
        failures = together([failures] + [{name: state[name][failed] for name in state} for state, failed in seen])
        if not args.keep_predictor:
            costly = amounts[:filled] > 0
            fitted, ended = fit(numbers[:filled], costly, trained['chosen strength'], start=fixed['predictor'], steps=100)
            fitted['mean_cost'] = np.float32(amounts[:filled][costly].mean())
            fixed = {'predictor': fitted, 'sizes': fixed['sizes']}
            record['predictor'] = {'rows': int(filled), 'costly share': float(costly.mean()), **ended}
        record['seconds'] = round(time.perf_counter() - started, 1)
        with open(out / 'checks.jsonl', 'a') as file:
            file.write(json.dumps(record) + '\n')
        print(f'check {check}, generation {generation}: reward {record["reward"]:.2f}, cost {record["cost"]:.1f} '
              f'(imagined shares: reward {imagined[0]:.2f}, cost {imagined[1]:.2f}); price now {price:.2f}, '
              f'{len(failures["deter"])} failure states kept', flush=True)

    # The final choice: the best checked adapters play more, each on the same seeds, and all their episodes count.
    finalists = best_of(checked, args.finalists)
    further = [next(seeds) for _ in range(args.more)]
    pooled = []
    for finalist in finalists:
        _, weights, with_it = adapters.load(out / finalist['adapter'], action_numbers, state_numbers)
        fixed = with_it
        seeds = iter(further)
        rewards, costs, _ = played(weights, args.more)
        rewards, costs = finalist['rewards'] + rewards, finalist['costs'] + costs
        pooled.append({'check': finalist['check'], 'adapter': finalist['adapter'], 'episodes': len(costs),
                       'reward': float(np.mean(rewards)), 'cost': float(np.mean(costs)),
                       'reward at its check': finalist['reward'], 'cost at its check': finalist['cost']})
        print(f'finalist of check {finalist["check"]}: reward {pooled[-1]["reward"]:.2f}, cost {pooled[-1]["cost"]:.2f} '
              f'over {len(costs)} episodes', flush=True)
    chosen, = best_of(pooled, 1)
    _, weights, with_it = adapters.load(out / chosen['adapter'], action_numbers, state_numbers)
    adapters.save(out / 'final.npz', adapter, weights, with_it)

    episodes_played = args.checks * args.episodes + args.finalists * args.more
    steps_an_episode = (len(first['cost']) - 1) * player.steps_per_decision
    (out / 'final.json').write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'task': player.task,
        'settings': vars(args),
        'adapter': adapter.describe(),
        'chosen': chosen,
        'within the cost limit': chosen['cost'] <= COST_LIMIT,
        'finalists': pooled,
        'generations': generation,
        'price at the end': price,
        'real episodes': {'behind the first predictor': len(began_with), 'played by the search': episodes_played},
        'simulator steps': (len(began_with) + episodes_played) * steps_an_episode,
        'cost run up': {'behind the first predictor': cost_behind_predictor, 'by the search': cost_paid},
    }, indent=2) + '\n')
    print(f'chosen: the adapter of check {chosen["check"]}, reward {chosen["reward"]:.2f}, cost {chosen["cost"]:.2f} over '
          f'{chosen["episodes"]} episodes; {(len(began_with) + episodes_played) * steps_an_episode:,} simulator steps used')


if __name__ == '__main__':
    main()
