"""Checks the look-ahead against the simulator: do the futures the model imagines really happen?

For a sample of decisions from the test episodes of the cost predictor, each of the 2A + 1
futures of the look-ahead is played for real. The simulator is started with the episode's
seed and stepped through the actions the episode recorded, up to the decision. From there
the future's action is held for the length of the look-ahead, and the cost and the reward
that really come are added up. No agent and no camera takes part: the world and its
physics follow from the seed and the actions alone.

The sample has two parts, both taken from decisions at which no cost is being charged:

    danger   the look-ahead expects holding the proposed action to cost half a unit or more
    quiet    it expects less

One thing must hold, or the check fails. Up to the decision, every replay has to give
exactly the cost and the reward the episode recorded. Otherwise the simulator was not in
the state the model was asked about.

Measured, for each look-ahead length in --decisions:

    futures   over every future played: how well the imagined cost tells the futures that
              really cost from those that did not (AUROC), how closely imagined and real
              follow each other for cost and for reward (correlation), and the imagined
              cost in total over the real. The proposed action's own future and the pushed
              ones are given apart, because the agent seldom holds a pushed action and the
              model has seen little of it.
    changes   over every push: the imagined change in cost against the real one, and when
              the model says a push lowers or raises the cost by half a unit or more, how
              often the real cost goes down, stays, or goes up.
    way out   over the danger decisions: the real cost of holding the proposed action, of
              the future the model expects to cost least, of a push taken blindly, and of
              the future that really cost least. With each, the range that holds 95% of
              the means when the episodes are redrawn with replacement.
    quiet     over the quiet decisions: the real cost of holding, and how often a push
              runs into a cost that the model did not expect.

From this folder:

  ../msCli/ms submit -t 3h -n check-look-ahead-real -- apptainer exec --nv --bind /datasets/smohan \\
      --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \\
      python3 check_look_ahead_real.py --predictor RUN_FOLDER/cost_predictor.npz --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np

from cost_predictor import auroc, read
from episodes import Player, states
from look_ahead import futures


def mean_and_range(values, episode, draws=1000):
    """The mean, and the range that holds 95% of the means when the episodes are redrawn with replacement."""
    random = np.random.default_rng(0)
    by_episode = [values[episode == number] for number in np.unique(episode)]
    means = [np.concatenate([by_episode[pick] for pick in random.integers(len(by_episode), size=len(by_episode))]).mean()
             for _ in range(draws)]
    return [float(values.mean())] + [float(end) for end in np.percentile(means, [2.5, 97.5])]


def together(first, second):
    """How closely two sets of numbers follow each other: 1 is exactly, 0 is not at all."""
    return float(np.corrcoef(np.ravel(first), np.ravel(second))[0, 1])


def directions(real_changes):
    """Of some real changes in cost: the shares that are down, none, and up."""
    return {'down': float((real_changes < 0).mean()), 'same': float((real_changes == 0).mean()),
            'up': float((real_changes > 0).mean()), 'cases': int(real_changes.size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictor', required=True,
                        help='a cost_predictor.npz, with the cost_predictor.json that was saved beside it')
    parser.add_argument('--decisions', type=int, nargs='+', default=[5, 10], help='the look-ahead lengths to try')
    parser.add_argument('--push', type=float, default=1.0,
                        help='how far a push goes towards the limit: 1 is all the way')
    parser.add_argument('--danger', type=int, default=200, help='how many danger decisions to play')
    parser.add_argument('--quiet', type=int, default=100, help='how many quiet decisions to play')
    parser.add_argument('--cameras', action='store_true',
                        help='draw the cameras while replaying, as when the episodes were played (slower)')
    args = parser.parse_args()

    # `files` are the episodes the predictor was scored on, which it has never been fitted to.
    predictor, trained, files = read(args.predictor, args.checkpoint)
    if trained['ahead'] != 0:
        raise SystemExit('the look-ahead needs a predictor of the cost of arriving in a state (--ahead 0)')
    for_gpu = {name: predictor[name] for name in ('weights', 'offset', 'mean_cost')}
    player = Player(args.checkpoint)
    frozen = player.frozen
    lengths = sorted(set(args.decisions))
    longest = lengths[-1]

    def compiled_futures(decisions):
        def imagine(state, proposed, predictor):
            return futures(frozen, predictor, state, proposed, decisions, args.push)
        return frozen.compiled(imagine)

    # First, what the model imagines at every entry of every test episode.
    imagine = {decisions: compiled_futures(decisions) for decisions in lengths}
    episodes, imagined = [], {decisions: [] for decisions in lengths}
    for file in files:
        saved = np.load(file)
        episodes.append({name: saved[name] for name in ('seed', 'action', 'proposed', 'reward', 'cost')})
        for decisions in lengths:
            imagined[decisions].append(imagine[decisions](states(saved), saved['proposed'], for_gpu))

    pick = np.random.default_rng(0)
    report, all_matched = {}, True
    for decisions in lengths:
        # The decisions that can be played: not charged, and with room for twice the longest look-ahead after them.
        danger, quiet = [], []
        for number, episode in enumerate(episodes):
            for entry in np.flatnonzero(episode['cost'][:len(episode['cost']) - 2 * longest] == 0):
                expects = imagined[decisions][number]['cost'][0, entry] >= 0.5
                (danger if expects else quiet).append((number, int(entry)))
        chosen = [(group, where) for group, pool, wanted in (('danger', danger, args.danger), ('quiet', quiet, args.quiet))
                  for where in pick.permutation(pool)[:wanted].tolist()]

        # Then each future of each chosen decision, played in the simulator.
        started = time.perf_counter()
        count = len(imagined[decisions][0]['cost'])
        real_cost, real_reward = np.zeros((len(chosen), count)), np.zeros((len(chosen), count))
        unmatched, reward_gap = 0, 0.0
        for row, (_, (number, entry)) in enumerate(chosen):
            episode = episodes[number]
            for future in range(count):
                held = imagined[decisions][number]['action'][future, entry]
                actions = np.concatenate([episode['action'][:entry], np.repeat(held[None], decisions, 0)])
                rewards, costs = player.replay(int(episode['seed']), actions, cameras=args.cameras)
                # Entry t of the replay is what the episode holds at entry t + 1.
                if len(costs) != len(actions) or not np.array_equal(costs[:entry], episode['cost'][1:entry + 1]):
                    unmatched += 1
                    continue
                reward_gap = max(reward_gap, float(np.abs(rewards[:entry] - episode['reward'][1:entry + 1]).max(initial=0)))
                real_cost[row, future], real_reward[row, future] = costs[entry:].sum(), rewards[entry:].sum()
            if (row + 1) % 50 == 0:
                print(f'look-ahead of {decisions}: {row + 1} of {len(chosen)} decisions played, '
                      f'{time.perf_counter() - started:.0f} s', flush=True)
        matched = unmatched == 0 and reward_gap < 1e-5
        all_matched &= matched

        group = np.array([name for name, _ in chosen])
        episode_of = np.array([number for _, (number, _) in chosen])
        entries = np.array([entry for _, (_, entry) in chosen])
        took = {name: np.stack([imagined[decisions][number][name][:, entry] for number, entry in zip(episode_of, entries)])
                for name in ('cost', 'reward')}
        imagined_cost, imagined_reward = took['cost'], took['reward']

        def agreement(columns):
            costs, really = imagined_cost[:, columns], real_cost[:, columns]
            return {
                'futures': int(really.size),
                'really cost': float((really > 0).mean()),
                'AUROC': auroc(costs.ravel(), really.ravel() > 0),
                'cost, imagined with real': together(costs, really),
                'cost, imagined over real': float(costs.sum() / really.sum()),
                'reward, imagined with real': together(imagined_reward[:, columns], real_reward[:, columns]),
            }

        imagined_change = imagined_cost[:, 1:] - imagined_cost[:, :1]
        real_change = real_cost[:, 1:] - real_cost[:, :1]
        there = group == 'danger'
        at = np.arange(there.sum())
        choice = imagined_cost[there].argmin(1)
        hindsight = real_cost[there].min(1)
        of_choice = real_cost[there][at, choice]
        holding = real_cost[there][:, 0]
        calm = group == 'quiet'
        surprise = real_cost[calm][:, 1:] > 0
        report[decisions] = {
            'played': {'danger': int(there.sum()), 'quiet': int(calm.sum()), 'futures each': count,
                       'replays that did not match': unmatched, 'largest reward gap before the decision': reward_gap,
                       'seconds': round(time.perf_counter() - started)},
            'futures': {'proposed action held': agreement(slice(0, 1)), 'pushed': agreement(slice(1, None))},
            'changes': {
                'cost change, imagined with real': together(imagined_change, real_change),
                'reward change, imagined with real': together(imagined_reward[:, 1:] - imagined_reward[:, :1],
                                                              real_reward[:, 1:] - real_reward[:, :1]),
                'model says 0.5 or more lower': directions(real_change[imagined_change <= -0.5]),
                'model says 0.5 or more higher': directions(real_change[imagined_change >= 0.5]),
            },
            'way out': {
                'holding really cost': float((holding > 0).mean()),
                'real cost, holding': mean_and_range(holding, episode_of[there]),
                'real cost, the future the model expects to cost least': mean_and_range(of_choice, episode_of[there]),
                'real cost, a push taken blindly': mean_and_range(real_cost[there][:, 1:].mean(1), episode_of[there]),
                'real cost, the future that really cost least': mean_and_range(hindsight, episode_of[there]),
                'saved by following the model': mean_and_range(holding - of_choice, episode_of[there]),
                'the model keeps the proposed action': float((choice == 0).mean()),
                'its choice against holding': directions(of_choice - holding),
                'real reward, holding': float(real_reward[there][:, 0].mean()),
                'real reward, the future the model expects to cost least': float(real_reward[there][at, choice].mean()),
            },
            'quiet': {
                'real cost, holding': float(real_cost[calm][:, 0].mean()),
                'holding really cost': float((real_cost[calm][:, 0] > 0).mean()),
                'pushes that really cost': float(surprise.mean()),
                'of those, the model expected 0.5 or more': float((imagined_cost[calm][:, 1:][surprise] >= 0.5).mean())
                if surprise.any() else None,
            },
        }

    print(f'\n{trained["task"]}, push {args.push:g}, cameras {"drawn" if args.cameras else "not drawn"}')
    for decisions, found in report.items():
        played, way_out, changes, calm = found['played'], found['way out'], found['changes'], found['quiet']
        print(f'\nlook-ahead of {decisions} decisions: {played["danger"]} danger and {played["quiet"]} quiet decisions, '
              f'{played["futures each"]} futures each, {played["seconds"]} s. '
              f'Replays that did not match: {played["replays that did not match"]}')
        for name, agrees in found['futures'].items():
            print(f'  futures, {name} ({agrees["futures"]}, {agrees["really cost"]:.0%} really cost): AUROC {agrees["AUROC"]:.3f}; '
                  f'cost, imagined with real {agrees["cost, imagined with real"]:.2f}, imagined over real '
                  f'{agrees["cost, imagined over real"]:.2f}; reward, imagined with real {agrees["reward, imagined with real"]:.2f}')
        print(f'  changes: cost, imagined with real {changes["cost change, imagined with real"]:.2f}; '
              f'reward {changes["reward change, imagined with real"]:.2f}')
        for says in ('model says 0.5 or more lower', 'model says 0.5 or more higher'):
            went = changes[says]
            print(f'    {says} ({went["cases"]} pushes): really down {went["down"]:.0%}, same {went["same"]:.0%}, up {went["up"]:.0%}')
        print(f'  way out: holding really cost at {way_out["holding really cost"]:.0%} of the danger decisions. Real cost, mean (95% range):')
        for name in ('holding', 'the future the model expects to cost least', 'a push taken blindly', 'the future that really cost least'):
            mean, low, high = way_out[f'real cost, {name}']
            print(f'    {name}: {mean:.2f} ({low:.2f} to {high:.2f})')
        mean, low, high = way_out['saved by following the model']
        against = way_out['its choice against holding']
        print(f'    saved by following the model: {mean:.2f} ({low:.2f} to {high:.2f}); its choice really cost less at '
              f'{against["down"]:.0%}, the same at {against["same"]:.0%}, more at {against["up"]:.0%}; '
              f'it keeps the proposed action at {way_out["the model keeps the proposed action"]:.0%}')
        print(f'    real reward: holding {way_out["real reward, holding"]:.3f}, the model\'s choice '
              f'{way_out["real reward, the future the model expects to cost least"]:.3f}')
        expected = calm['of those, the model expected 0.5 or more']
        print(f'  quiet: holding really cost at {calm["holding really cost"]:.0%} (mean {calm["real cost, holding"]:.2f}); '
              f'a push really cost at {calm["pushes that really cost"]:.0%}'
              + ('' if expected is None else f', and the model expected 0.5 or more for {expected:.0%} of those'))

    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'check_look_ahead_real.json'
    out.write_text(json.dumps({
        'checkpoint': trained['checkpoint'],
        'task': trained['task'],
        'predictor': args.predictor,
        'test episodes': trained['files']['test'],
        'push': args.push,
        'cameras drawn': args.cameras,
        'every replay matched the episode up to the decision': bool(all_matched),
        'look-ahead of': {str(decisions): found for decisions, found in report.items()},
    }, indent=2) + '\n')
    if not all_matched:
        raise SystemExit('FAIL: some replays did not give what the episode recorded up to the decision')
    print('\nPASS: every replay matched its episode up to the decision; the rest above is measured, not judged')


if __name__ == '__main__':
    main()
