"""Scores an adapter inside the frozen world model, with no simulator.

From a set of starting states the world model imagines what follows when the agent acts
with the adapter (brainStorming/adapterRecipeDesign.md, decision 7). One imagined decision
is a real decision with the world model in the simulator's place:

    the frozen agent proposes an action in the imagined state
    the adapter gives the action to take
    the world model gives the state that follows
    the cost predictor and the reward predictor say what arriving there costs and earns

The starting states are states the unadapted agent really was in, taken from saved
episodes by starting_states(). An Imagination holds them for one kind of adapter, and its
roll() imagines a number of decisions from every one of them for a row of weights.

Why it is built this way:

  The proposed action and the next state are random draws, as when the agent plays and as
  the world model was trained. check_cost_ahead.py found that the most likely next state,
  taken again and again, makes the model expect too much cost. The most likely ones can
  still be asked for, to compare. Inside the adapter the look-ahead keeps its most likely
  roll either way, because an adapter has to work here exactly as it does when playing.

  Every candidate is rolled from the same starting states with the same random numbers:
  those of a roll all follow from one seed. Two rows of weights then differ in their rolls
  only because the weights differ. At zero weights the roll is the unadapted agent's with
  those random numbers, which is what a candidate is measured against.

  One imagined decision is compiled and the roll is a plain loop over it. A roll compiled
  whole would have every look-ahead of every decision written out in it, hundreds of
  steps of the world model, and would take minutes to compile.
"""

import numpy as np

from episodes import states
from look_ahead import expected


def starting_states(files, count, room, seed=0):
    """Draws `count` decisions at random from saved episodes, as places for imagined rolls to start.

    Only decisions with `room` more decisions after them are drawn, so that what really
    followed each one is known.

    Returns, with one row per starting state:

        state           the internal states, in the form the pieces of frozen_agent.py take
        episode         which of `files` it came from, and `entry` which decision of that episode
        charged         whether a cost was being charged at it
        cost, reward    what really followed: one column for each of the `room` decisions after it
    """
    saved = [np.load(file) for file in files]
    places = [(number, entry) for number, episode in enumerate(saved)
              for entry in range(len(episode['cost']) - room)]
    random = np.random.default_rng(seed)
    chosen = np.array(sorted(places[pick] for pick in random.choice(len(places), count, replace=False)))
    drawn = {name: [] for name in ('deter', 'stoch', 'charged', 'cost', 'reward')}
    for number, episode in enumerate(saved):
        entries = chosen[chosen[:, 0] == number, 1]
        state = states(episode, entries)
        cost, reward = episode['cost'], episode['reward']
        # Entry t + 1 of an episode holds what came of the action of entry t.
        after = entries[:, None] + 1 + np.arange(room)
        for name, part in (('deter', state['deter']), ('stoch', state['stoch']), ('charged', cost[entries] > 0),
                           ('cost', cost[after]), ('reward', reward[after])):
            drawn[name].append(part)
    drawn = {name: np.concatenate(parts) for name, parts in drawn.items()}
    return {'state': {'deter': drawn['deter'], 'stoch': drawn['stoch']}, 'episode': chosen[:, 0], 'entry': chosen[:, 1],
            'charged': drawn['charged'], 'cost': drawn['cost'], 'reward': drawn['reward']}


class Imagination:
    """Imagined rolls for one kind of adapter, all from the same starting states.

    frozen     the frozen agent, a FrozenDreamer or a FrozenTDMPC2
    adapter    an adapter.Adapter or an adapter.MemoryAdapter
    fixed      what the adapter needs besides its weights: the cost predictor and the sizes
    state      the starting states
    sample     True: the proposed action and the next state are random draws.
               False: they are the most likely ones, and nothing in a roll is random
    """

    def __init__(self, frozen, adapter, fixed, state, sample=True):
        remembers = adapter.form == 'memory'

        def decision(state, memory, fixed, weights):
            xp = frozen.xp
            state = {name: state[name] for name in frozen.names}
            proposed = frozen.propose(state, sample=sample)
            if remembers:
                made = adapter.decide(frozen, fixed, weights, state, proposed, memory)
                action, memory = made['action'], made['memory']
            else:
                action = adapter.decide(frozen, fixed, weights, state, proposed)['action']
            following, reward = frozen.step(state, action, sample=sample)
            # The simulator and the world model clip an action, so it is the clipped ones that are compared.
            taken, offered = xp.clip(frozen.as_float(action), -1, 1), xp.clip(frozen.as_float(proposed), -1, 1)
            return {
                'state': following,
                'memory': memory,
                'cost': expected(xp, fixed['predictor'], frozen.numbers(following)),
                'reward': reward,
                'value': frozen.worth(following),
                'moved': (taken != offered).any(-1),
                'change': xp.abs(taken - offered).mean(-1),
                # How large the action taken is and how large the proposed one was, for the share of its size that is kept.
                'size taken': (taken ** 2).sum(-1) ** 0.5,
                'size proposed': (offered ** 2).sum(-1) ** 0.5,
            }

        self._frozen, self._adapter, self._fixed, self._state = frozen, adapter, fixed, state
        self._decision = frozen.compiled(decision)

    def roll(self, weights, decisions, seed=0, state=None, fixed=None, memory=None):
        """Imagines `decisions` decisions from every starting state, with the adapter at `weights`.

        `state` and `fixed`, if given, take the place of the starting states and of the
        cost predictor and sizes that this Imagination was made with. A search uses them
        to start each generation somewhere new and to bring in a retrained predictor.
        Starting states of the same number as before need no new compiling. `memory` is
        the memory each roll starts with, one row per starting state; left out, every
        roll starts with an empty one.

        Returns arrays with one row per decision and one column per starting state:

            cost            the expected cost of arriving in the state that follows the decision
            reward          the reward predicted for arriving there
            value           the agent's own estimate of the reward still to come from there
            moved           whether the adapter changed the action
            change          how far the action taken lies from the proposed one, as a mean
                            over the action's numbers
            size taken      the length of the action taken, and
            size proposed   of the action the agent proposed
        """
        # Each imagined decision takes its random numbers from the frozen agent's own
        # source for adapters, which this sets to the same place for every roll.
        self._frozen.reseed(seed)
        state = self._state if state is None else state
        fixed = self._fixed if fixed is None else fixed
        states = len(next(iter(state.values())))
        if memory is None:
            memory = self._adapter.start(states) if self._adapter.form == 'memory' else np.zeros((states, 1), np.float32)
        kept = {name: [] for name in ('cost', 'reward', 'value', 'moved', 'change', 'size taken', 'size proposed')}
        for _ in range(decisions):
            came = self._decision(state, memory, fixed, weights)
            state, memory = came['state'], came['memory']
            for name, rows in kept.items():
                rows.append(came[name])
        return {name: np.stack(rows) for name, rows in kept.items()}
