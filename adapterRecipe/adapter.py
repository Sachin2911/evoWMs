"""The adapter: a small set of weights that changes the action the frozen agent proposes.

An adapter sees one decision at a time. It is given the agent's internal state and the
action the agent proposed, and gives back the action to take. It never changes the agent,
and with every weight at zero it gives back the proposed action untouched
(brainStorming/adapterRecipeDesign.md, decisions 3 and 4).

Its form is how it acts:

    residual   It works out a gain and a shift for every action number and takes
                   proposed x (1 + gain) + shift
               clipped to the limits of the action.
    re-rank    It takes the proposed action and the pushed versions of it that the
               look-ahead imagines, and takes the one with the highest score. The proposed
               action scores 0 and is kept unless another scores higher.
    mix        A re-rank in which the action of a residual is one more candidate.

Its size is how a residual gets its gains and shifts:

    tens        a weighted sum of the look-ahead numbers
    hundreds    a small network on the look-ahead numbers, with one hidden layer
    thousands   a weighted sum of a few mixtures of the raw internal state. Which mixtures
                is part of the weights (a low-rank map)

The score of a candidate in a re-rank or a mix is

    reward weight x its reward change  -  cost weight x its cost change

where both changes are against holding the proposed action and are divided by the sizes
that look_ahead.sizes() measured. These two weights are all the weights a re-rank has, and
their ratio is the price of cost.

The weights come as one row, which is what a search works with. With A action numbers and
S state numbers:

    residual, tens        (4A + 2) x 2A                    40 for A = 2
    residual, hundreds    16 x (4A + 2) + 2A x 17         228
    residual, thousands   S + 2A + 2A                   2,824 for S = 2,816
    re-rank               2
    mix                   the residual's, then the 2 of the re-rank

In a residual of size tens the row is a table with one line per output, first the gains and
then the shifts. A line has one weight per look-ahead number, in the order of
look_ahead.names(), and a last weight that is added on.

Why it is built this way:

  The agent's action numbers can lie beyond the limits, because it draws them from a bell
  curve. The simulator and the world model both clip them, so the adapter works with the
  clipped action. But where it leaves an action number where it was, it hands back the
  agent's own number, unclipped. The world model clips by a multiplication that can be off
  in the last digit, so a clipped copy would not be exactly what the unadapted agent hands
  over, and an episode would no longer replay exactly.

  A gain and a shift have no bound of their own. The limits of the action are the only
  bound, which leaves no value to set. check_look_ahead_real.py found the world model about
  as right about an action pushed to its limit as about the proposed one.

  The look-ahead numbers are divided by sizes measured on the unadapted agent's episodes,
  and nothing is subtracted from them. A number that is zero stays zero, so an adapter of
  size tens whose added-on weights are zero does nothing where no push changes anything.
  The raw state is not scaled: its numbers already lie between -1 and 1.

  A re-rank chooses among the pushed actions and not among further actions from the
  agent. Those lie too close to the proposed one to hold a way out (design, decision 3),
  and the pushed ones are already imagined, so choosing among them costs no further roll.
"""

import numpy as np

from look_ahead import LENGTHS, futures, held, hold, hold_by_stretch, names, names_by_stretch, numbers_by_stretch, numbers_of, squashed

FORMS = ('residual', 're-rank', 'mix')
SIZES = ('tens', 'hundreds', 'thousands')


def steering(action_numbers):
    """The weights of steering, the adapter written by hand: a residual of size tens.

    Each action number is shifted towards the push that the look-ahead expects to cost
    less, by the difference between the two cost changes. Every other weight is zero.
    """
    called = names(action_numbers)
    # A line per output, the gains first and then the shifts; a column per look-ahead number and one that is added on.
    table = np.zeros((2 * action_numbers, len(called) + 1), np.float32)
    for number in range(action_numbers):
        table[action_numbers + number, called.index(f'cost change, number {number} up')] = -1
        table[action_numbers + number, called.index(f'cost change, number {number} down')] = 1
    return table.ravel()


def save(path, adapter, weights, fixed):
    """Saves an adapter with its weights and with what it needs besides them, as one .npz file."""
    settings = {'form': adapter.form, 'weights': np.asarray(weights, np.float32), 'sizes': fixed['sizes']}
    if adapter.form == 'memory':
        settings.update(lengths=np.array(adapter.lengths), memory=adapter.memory, added_on=adapter.added_on, stop=adapter.stop,
                        roll=adapter.roll)
    else:
        settings.update(size=adapter.size or '', decisions=adapter.decisions, push=adapter.push,
                        hidden=adapter.hidden, mixtures=adapter.mixtures)
    # Every array of the cost predictor is saved under its own name, with predictor_ in front,
    # and every array of the tail, if the adapter reads one, with tail_ in front.
    np.savez(path, **settings, **{f'predictor_{name}': value for name, value in fixed['predictor'].items()},
             **{f'tail_{name}': value for name, value in (fixed.get('tail') or {}).items()})


def load(path, action_numbers, state_numbers=None):
    """Reads what save() wrote: the adapter, its weights, and the cost predictor and sizes it works with.

    An adapter that was saved with a tail (tail.py) gets it back as `tail`.
    """
    saved = np.load(path)
    if str(saved['form']) == 'memory':
        adapter = MemoryAdapter(action_numbers, tuple(int(length) for length in saved['lengths']),
                                bool(saved['memory']), bool(saved['added_on']), bool(saved['stop']),
                                str(saved['roll']) if 'roll' in saved.files else 'likely')
    else:
        adapter = Adapter(str(saved['form']), str(saved['size']) or None, action_numbers, state_numbers,
                          int(saved['decisions']), float(saved['push']), int(saved['hidden']), int(saved['mixtures']))
    predictor = {name[len('predictor_'):]: saved[name] for name in saved.files if name.startswith('predictor_')}
    fixed = {'predictor': predictor, 'sizes': saved['sizes']}
    tail = {name[len('tail_'):]: saved[name] for name in saved.files if name.startswith('tail_')}
    if tail:
        fixed['tail'] = tail
    return adapter, saved['weights'], fixed


class Adapter:
    """An adapter without its weights: its form, its size and its settings.

    form             'residual', 're-rank' or 'mix'
    size             'tens', 'hundreds' or 'thousands'; None for a re-rank, which has no residual
    action_numbers   how many numbers an action has (A)
    state_numbers    how many numbers FrozenDreamer.numbers gives for a state (S)
    decisions        how many decisions the imagined futures last
    push             how far the look-ahead pushes a number towards its limit
    hidden           the hidden units of size hundreds
    mixtures         the mixtures of the state that size thousands uses

    `count` is the number of weights.
    """

    def __init__(self, form, size, action_numbers, state_numbers, decisions, push=1.0, hidden=16, mixtures=1):
        if form not in FORMS or size not in ((None,) if form == 're-rank' else SIZES):
            raise ValueError(f'there is no adapter of form {form} and size {size}')
        self.form, self.size = form, size
        self.action_numbers = action_numbers
        self.decisions, self.push = decisions, push
        self.hidden, self.mixtures = hidden, mixtures
        # What a residual reads, and how many weights turn that into a gain and a shift per action number.
        outputs = 2 * action_numbers
        look_ahead_numbers = 4 * action_numbers + 1
        self._inputs = {None: 0, 'tens': look_ahead_numbers, 'hundreds': look_ahead_numbers,
                        'thousands': state_numbers}[size]
        self._in_residual = {
            None: 0,
            'tens': (self._inputs + 1) * outputs,
            'hundreds': hidden * (self._inputs + 1) + outputs * (hidden + 1),
            'thousands': mixtures * (self._inputs + outputs) + outputs,
        }[size]
        self.count = self._in_residual + (0 if form == 'residual' else 2)

    def describe(self):
        """The settings and the number of weights, for the record of a run."""
        settings = {'form': self.form, 'size': self.size, 'weights': self.count, 'decisions': self.decisions}
        if self.size != 'thousands' or self.form != 'residual':
            settings['push'] = self.push
        if self.size == 'hundreds':
            settings['hidden units'] = self.hidden
        if self.size == 'thousands':
            settings['mixtures'] = self.mixtures
        return settings

    def _gains_and_shifts(self, weights, inputs):
        """What a residual's weights make of its inputs: a gain and a shift per action number, for every state."""
        import jax.numpy as jnp

        outputs = 2 * self.action_numbers
        if self.size == 'tens':
            table = weights.reshape(outputs, self._inputs + 1)
            out = inputs @ table[:, :-1].T + table[:, -1]
        elif self.size == 'hundreds':
            first, second = jnp.split(weights, [self.hidden * (self._inputs + 1)])
            first = first.reshape(self.hidden, self._inputs + 1)
            second = second.reshape(outputs, self.hidden + 1)
            # tanh keeps each hidden unit between -1 and 1, and gives 0 for 0.
            hidden = jnp.tanh(inputs @ first[:, :-1].T + first[:, -1])
            out = hidden @ second[:, :-1].T + second[:, -1]
        else:
            mixing, spreading, added = jnp.split(
                weights, [self.mixtures * self._inputs, self.mixtures * (self._inputs + outputs)])
            mixed = inputs @ mixing.reshape(self.mixtures, self._inputs).T
            out = mixed @ spreading.reshape(outputs, self.mixtures).T + added
        return out[:, :self.action_numbers], out[:, self.action_numbers:]

    def decide(self, frozen, fixed, weights, state, proposed):
        """The action to take in every state of a batch, with what was worked out on the way.

        It is written with the pieces of frozen_agent.py, so it has to be called from a
        function that was passed to FrozenDreamer.compiled; compiled() below does that.

        fixed      what is measured and never searched: `predictor`, the cost predictor's
                   weights, offset and mean_cost, and `sizes`, from look_ahead.sizes()
        weights    the adapter's `count` weights, in one row
        state      the internal states, one row each
        proposed   the action the frozen agent proposed in each

        Returns `action`. A re-rank or a mix also returns, with one row per candidate and
        one column per state:

            candidates   the actions it chose among. Row 0 is the proposed action, then
                         come the pushed ones in the order of look_ahead.pushed(), and in
                         a mix the residual's is last
            cost         the expected cost of holding each candidate
            reward       the reward of holding it, with the value estimate of the last state
            chosen       which candidate was taken in each state
        """
        import jax.numpy as jnp

        if weights.shape != (self.count,):
            raise ValueError(f'this adapter has {self.count} weights, and was given an array of shape {weights.shape}')
        predictor, sizes = fixed['predictor'], fixed['sizes']
        state = {name: state[name] for name in ('deter', 'stoch')}
        proposed = proposed.astype(jnp.float32)
        held = jnp.clip(proposed, -1, 1)
        # The futures of the look-ahead. Only a residual on the raw state does without them.
        looks_ahead = self.form != 'residual' or self.size != 'thousands'
        imagined = futures(frozen, predictor, state, proposed, self.decisions, self.push) if looks_ahead else None

        if self.form != 're-rank':
            inputs = frozen.numbers(state) if self.size == 'thousands' else numbers_of(imagined) / sizes
            gain, shift = self._gains_and_shifts(weights[:self._in_residual], inputs)
            moved = jnp.clip(held * (1 + gain) + shift, -1, 1)
            # A number that is left where it was is handed back as the agent gave it.
            residual = jnp.where(moved == held, proposed, moved)
            if self.form == 'residual':
                return {'action': residual}

        # The candidates: the proposed action as the agent gave it, then the pushed ones.
        candidates = jnp.concatenate([proposed[None], imagined['action'][1:]])
        cost = imagined['cost']
        # The value estimate stands for the reward still to come after the last imagined decision.
        reward = imagined['reward'] + imagined['value']
        if self.form == 'mix':
            own = hold(frozen, predictor, state, jnp.clip(residual, -1, 1)[None], self.decisions)
            candidates = jnp.concatenate([candidates, residual[None]])
            cost = jnp.concatenate([cost, own['cost']])
            reward = jnp.concatenate([reward, own['reward'] + own['value']])
        reward_weight, cost_weight = weights[-2], weights[-1]
        # sizes[1] is the size of a cost change and sizes[-1] that of a reward change.
        score = reward_weight * (reward - reward[:1]) / sizes[-1] - cost_weight * (cost - cost[:1]) / sizes[1]
        # Of equal scores the first is taken, so the proposed action stays unless another scores higher.
        chosen = jnp.argmax(score, 0)
        action = candidates[chosen, jnp.arange(chosen.shape[0])]
        return {'action': action, 'candidates': candidates, 'cost': cost, 'reward': reward, 'chosen': chosen}

    def compiled(self, frozen, everything=False):
        """This adapter as a function of plain arrays, called as (state, proposed, fixed, weights).

        It gives the action to take, or with `everything` all that decide() returns. With
        the weights filled in, it is an adapter as Player.play takes one:

            act = adapter.compiled(frozen)
            player.play(seeds, adapter=lambda state, proposed: act(state, proposed, fixed, weights))
        """
        def adapt(state, proposed, fixed, weights):
            worked_out = self.decide(frozen, fixed, weights, state, proposed)
            return worked_out if everything else worked_out['action']

        return frozen.compiled(adapt)


class MemoryAdapter:
    """The adapter of implementation/adapterArchitecture.md: a gain and a shift per action number, with memory.

    At every decision it reads the look-ahead numbers by stretch (look_ahead.py), squashed,
    and a memory of its own last outputs. One layer turns them into a gain and a shift for
    every action number, each between -1 and 1, and the action taken is

        proposed x (1 + gain) + shift

    clipped to the limits of the action. The memory is a fading average of the outputs,
    one number per output, and how fast each fades is one of the weights.

    action_numbers   how many numbers an action has (A)
    lengths          where the stretches of the look-ahead end, in decisions
    memory           False leaves the memory out, to see what it adds
    added_on         True gives every output a weight that is added on whatever the inputs
                     are, to see what leaving those out costs
    stop             False leaves the stopped action out of the look-ahead
    roll             how the look-ahead imagines: 'likely' takes the most likely class of
                     every categorical choice at every step, and 'mean' writes each
                     choice as how likely every class is. The second has no jumps

    The weights come as one row of `count` numbers: a table with one line per output, the
    gains first and then the shifts, and after it the fade weights. A line has one weight
    per look-ahead number in the order of look_ahead.names_by_stretch(), then one per
    memory number, then the added-on weight if there is one. For two action numbers that
    is 4 x (33 + 4) + 4 = 152 weights, and for four 528.

    With every weight at zero the output is zero, the memory stays zero, and the action
    is the proposed one, handed back as the agent gave it.

    It is written with the frozen agent's own array functions (frozen.xp), so the same
    lines run on DreamerV3 and on TD-MPC2.
    """

    form = 'memory'
    size = None

    def __init__(self, action_numbers, lengths=LENGTHS, memory=True, added_on=False, stop=True, roll='likely'):
        if roll not in ('likely', 'mean'):
            raise ValueError(f'the look-ahead rolls on the likely classes or on the mean, not on {roll}')
        self.action_numbers = action_numbers
        self.lengths = tuple(int(length) for length in lengths)
        self.memory, self.added_on, self.stop, self.roll = bool(memory), bool(added_on), bool(stop), roll
        self.outputs = 2 * action_numbers
        self.inputs = len(self.lengths) * (4 * action_numbers + (3 if self.stop else 1))
        self.remembered = self.outputs if self.memory else 0
        self._columns = self.inputs + self.remembered + int(self.added_on)
        self._in_table = self.outputs * self._columns
        self.count = self._in_table + self.remembered

    def describe(self):
        """The settings and the number of weights, for the record of a run."""
        return {'form': self.form, 'weights': self.count, 'lengths': list(self.lengths), 'memory': self.memory,
                'added-on weights': self.added_on, 'stopped action': self.stop, 'look-ahead roll': self.roll}

    def names(self):
        """What each column of the table reads, in order."""
        called = names_by_stretch(self.action_numbers, self.lengths, self.stop)
        called += [f'memory of {kind} {number}' for kind in ('gain', 'shift') for number in range(self.action_numbers)]
        return called[:self.inputs + self.remembered] + (['added on'] if self.added_on else [])

    def table(self, weights):
        """The table part of a row of weights: one line per output, one column per thing read."""
        return np.asarray(weights)[:self._in_table].reshape(self.outputs, self._columns)

    def start(self, states=1):
        """The memory at the start of an episode, for `states` states. It has one column even with no memory, which is never read."""
        return np.zeros((states, max(self.remembered, 1)), np.float32)

    def steering(self, scale=1.0):
        """Weights that steer by hand in this form: each action number is shifted towards the push expected to cost less.

        It reads the cost changes of the two pushes in every stretch and uses nothing
        else: no gain, no reward, no stopped action and no memory.
        """
        called = self.names()
        table = np.zeros((self.outputs, self._columns), np.float32)
        for number in range(self.action_numbers):
            for column, name in enumerate(called):
                if name.startswith(f'cost change, number {number} up'):
                    table[self.action_numbers + number, column] = -scale
                if name.startswith(f'cost change, number {number} down'):
                    table[self.action_numbers + number, column] = scale
        return np.concatenate([table.ravel(), np.zeros(self.remembered, np.float32)])

    def decide(self, frozen, fixed, weights, state, proposed, memory, strength=1.0):
        """The action to take in every state of a batch, and the memory to carry to the next decision.

        fixed      what is measured and never searched: `predictor`, the cost predictor,
                   and `sizes`, from look_ahead.sizes_by_stretch(). With `tail`, a
                   predictor of the cost to come (tail.py), every cost the look-ahead
                   reads has it added, and the sizes have to be measured with it too
        weights    the adapter's `count` weights, in one row
        state      the internal states, one row each
        proposed   the action the frozen agent proposed in each
        memory     the memory at this decision, one row per state; start() gives the first
        strength   what the gains and shifts are multiplied by before they are applied to
                   the action. It is a plain number, set when the adapter is played and
                   never searched. The memory and `output` keep the gains and shifts as
                   the layer gave them

        Returns `action`, `memory` for the next decision, and `output`: the gains and
        then the shifts.
        """
        xp = frozen.xp
        if tuple(weights.shape) != (self.count,):
            raise ValueError(f'this adapter has {self.count} weights, and was given an array of shape {tuple(weights.shape)}')
        state = {name: state[name] for name in frozen.names}
        proposed = frozen.as_float(proposed)
        clipped = xp.clip(proposed, -1, 1)
        imagined = hold_by_stretch(frozen, fixed['predictor'], state, held(frozen, proposed, self.stop), self.lengths, self.roll,
                                   fixed.get('tail'))
        inputs = squashed(xp, numbers_by_stretch(frozen, imagined), fixed['sizes'])

        table = weights[:self._in_table].reshape(self.outputs, self._columns)
        summed = inputs @ table[:, :self.inputs].T
        if self.remembered:
            summed = summed + memory @ table[:, self.inputs:self.inputs + self.remembered].T
        if self.added_on:
            summed = summed + table[:, -1]
        # tanh is the ceiling inside the loop: without it a memory that feeds on itself grows without limit.
        output = xp.tanh(summed)
        gain, shift = output[:, :self.action_numbers], output[:, self.action_numbers:]
        if strength != 1.0:
            # The output is turned up or down after the tanh. Larger weights would not do the same: the tanh would cut them off.
            gain, shift = strength * gain, strength * shift
        moved = xp.clip(clipped * (1 + gain) + shift, -1, 1)
        # A number that is left where it was is handed back as the agent gave it.
        action = xp.where(moved == clipped, proposed, moved)
        if self.remembered:
            fade = 1 / (1 + xp.exp(-weights[self._in_table:]))
            memory = (1 - fade) * memory + fade * output
        return {'action': action, 'memory': memory, 'output': output}

    def compiled(self, frozen, everything=False, strength=1.0):
        """This adapter as a function of plain arrays, called as (state, proposed, memory, fixed, weights).

        It gives the action to take and the next memory, or with `everything` all that
        decide() returns. `strength` is that of decide(), fixed for every call.
        Player.play takes it with the weights filled in:

            act = adapter.compiled(frozen)
            player.play(seeds, adapter=lambda state, proposed, memory: act(state, proposed, memory, fixed, weights),
                        memory=adapter.start())
        """
        def adapt(state, proposed, memory, fixed, weights):
            worked_out = self.decide(frozen, fixed, weights, state, proposed, memory, strength)
            return worked_out if everything else (worked_out['action'], worked_out['memory'])

        return frozen.compiled(adapt)
