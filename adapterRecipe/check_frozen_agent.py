"""Checks frozen_agent.py and episodes.py: an adapter that changes nothing must replay the unadapted agent exactly.

Plays the same few episodes four ways and compares every action, reward and cost:

  alone        the agent called directly, all episodes in one go, as
               safeDreamerBaselines/evaluate.py plays
  untouched    through Player.play with no adapter, which is how episodes.py collects
  zero         through Player.play with an adapter that uses all five pieces and then
               multiplies what they gave by zero
  alone again  the first way once more

All four must agree. If the two `alone` runs differ from each other, episodes do not replay
in this job at all and the check says nothing about the wrapper.

In the `zero` run the pieces are also checked at every decision for the right shapes and
for finite numbers, and a few of their numbers are printed next to what really happened, so
that a wrong network would show.

Run it from this folder, with the cameras drawn on the CPU (MUJOCO_GL=osmesa). With the
cameras on the GPU an episode does not always replay exactly.

  ../msCli/ms submit -t 30m -n check-frozen -- apptainer exec --nv --bind /datasets/smohan \\
      --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \\
      python3 check_frozen_agent.py --checkpoint \\
      /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
"""

import argparse
import json
import os
import pathlib

import numpy as np

from episodes import Player


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--first-seed', type=int, default=0)
    args = parser.parse_args()
    seeds = range(args.first_seed, args.first_seed + args.episodes)

    player = Player(args.checkpoint)
    agent, frozen = player.agent, player.frozen

    def pieces(state, proposed):
        """Every piece, used once."""
        following = frozen.roll_forward(state, proposed)
        return {
            'numbers': frozen.numbers(state),
            'reward here': frozen.reward(state),
            'reward next': frozen.reward(following),
            'value': frozen.value(state),
            'likely action': frozen.propose(state, sample=False),
            'drawn action': frozen.propose(state),
        }

    use_pieces = frozen.compiled(pieces)
    seen = []    # what the pieces gave at each decision of the `zero` run

    def zero(state, proposed):
        gave = use_pieces(state, proposed)
        as_numbers = np.concatenate([state['deter'], state['stoch'].reshape(1, -1)], -1)
        if not np.array_equal(gave['numbers'], as_numbers.astype(np.float32)):
            raise SystemExit('numbers() does not give back the state it was handed')
        shapes = {'reward here': (1,), 'reward next': (1,), 'value': (1,),
                  'likely action': proposed.shape, 'drawn action': proposed.shape}
        for name, shape in shapes.items():
            if gave[name].shape != shape or not np.isfinite(gave[name]).all():
                raise SystemExit(f'{name} is wrong: shape {gave[name].shape}, values {gave[name]}')
        seen.append({**{name: gave[name][0] for name in shapes}, 'taken action': proposed[0]})
        # Zero weights: what the pieces gave is multiplied by zero before it meets the action.
        gain = 0.0 * gave['reward next'][:, None]
        shift = 0.0 * gave['likely action']
        return proposed * (1 + gain) + shift

    def alone(*inputs):
        return agent.policy(*inputs, mode='eval')

    # `alone` plays all the episodes in one go and play() starts afresh for each one, so
    # their agreement also shows that the two ways of starting an episode are the same.
    played = {
        'alone': player.run(alone, seeds),
        'untouched': list(player.play(seeds)),
        'zero': list(player.play(seeds, adapter=zero)),
        'alone again': player.run(alone, seeds),
    }

    def same(first, second):
        return all(np.array_equal(a[name], b[name])
                   for a, b in zip(played[first], played[second]) for name in ('action', 'reward', 'cost'))

    for name, episodes in played.items():
        totals = ', '.join(f'reward {e["reward"].sum():.3f} cost {e["cost"].sum():.0f}' for e in episodes)
        print(f'{name:12s} {totals}')

    # The pieces next to what really happened in the `zero` run, one line per episode.
    # Entry t of an episode is the decision made on seeing observation t.
    piece_report, start = [], 0
    for episode in played['zero']:
        gave = seen[start:start + len(episode['reward'])]
        start += len(gave)
        gave = {name: np.array([entry[name] for entry in gave], np.float64) for name in gave[0]}
        real = episode['reward'].astype(np.float64)
        piece_report.append({
            'real reward': real.sum(),
            # reward(state) is the reward for arriving in the state the agent is in now.
            'reward here, summed': gave['reward here'].sum(),
            # reward(roll_forward(state, action)) is the model's guess one decision ahead.
            'reward next, summed': gave['reward next'][:-1].sum(),
            'reward here against real, correlation': np.corrcoef(gave['reward here'], real)[0, 1],
            'reward next against real, correlation': np.corrcoef(gave['reward next'][:-1], real[1:])[0, 1],
            'value, mean': gave['value'].mean(),
            'taken action minus likely action, spread': (gave['taken action'] - gave['likely action']).std(),
            'drawn action minus likely action, spread': (gave['drawn action'] - gave['likely action']).std(),
        })
    for number, report in enumerate(piece_report):
        print(f'pieces, episode {number}: ' + ', '.join(f'{name} {value:.3f}' for name, value in report.items()))

    replays = same('alone', 'alone again')
    verdicts = {
        'episodes replay at all (alone = alone again)': replays,
        'play() with no adapter replays': same('alone', 'untouched'),
        'zero-weight adapter that uses the pieces replays': same('alone', 'zero'),
    }
    for name, verdict in verdicts.items():
        print(f'{name}: {verdict}')
    out = pathlib.Path(os.environ.get('MS_RUN_DIR', '.')) / 'check_frozen_agent.json'
    out.write_text(json.dumps({
        'checkpoint': pathlib.Path(args.checkpoint).name,
        'seeds': list(seeds),
        'rendering': os.environ.get('MUJOCO_GL'),
        'verdicts': verdicts,
        'totals': {name: [{'reward': float(e['reward'].sum()), 'cost': float(e['cost'].sum())}
                          for e in episodes] for name, episodes in played.items()},
        'pieces': [{name: float(value) for name, value in report.items()} for report in piece_report],
    }, indent=2) + '\n')

    if not replays:
        raise SystemExit('INCONCLUSIVE: the unadapted agent did not replay itself in this job')
    if not all(verdicts.values()):
        raise SystemExit('FAIL')
    print('PASS')


if __name__ == '__main__':
    main()
