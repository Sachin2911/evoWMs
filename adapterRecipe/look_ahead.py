"""The look-ahead: what the frozen world model expects if the proposed action, or a pushed version of it, is held.

This is the main thing an adapter reads (brainStorming/adapterRecipeDesign.md, decision 2).
For a state and the action the frozen agent proposed in it, the model imagines 2A + 1
short futures, where A is how many numbers an action has:

    the proposed action, held for `decisions` decisions
    for each action number in turn: the same with that number pushed up, then pushed down

Along each future it adds up the expected cost (from the cost predictor) and the reward
(from the model's reward predictor), and to the reward it adds the agent's value estimate
of the last state, for the reward still to come after it. The adapter gets 4A + 1 numbers:

    the expected cost of holding the proposed action           1
    how much each push changes the expected cost               2A
    how much each push changes the reward                      2A

names() gives their order. pushed() gives the actions that are held, hold() imagines holding
any actions, and sizes() measures how large the numbers usually are.

Why it is built this way:

  The action is held, and the agent is not asked again on the way. An adapter acts at
  every decision, so a push that the agent undoes one decision later is not what will
  happen. Holding stands in for an adapter that keeps pushing. The other way, a short push
  after which the agent carries on, is still to be measured against this one.

  A push lasts the whole future. check_cost_ahead.py showed that which action is taken
  makes little difference to the cost of the next few decisions and a large one further
  out, so a push for a single decision would show next to nothing.

  A push is towards the limit of the action number. Both agents keep every action number
  between -1 and 1. A number already at a limit cannot be pushed further that way, and its
  two changes for that push are then exactly zero.

  The model's most likely next state is used at every step, not a random draw. The same
  state and action then always give the same numbers, and two futures differ only because
  their actions differ.

  Nothing is discounted. Over the lengths used here the agent's own discount would change
  the sums by a few percent, and by the same few percent for every future.
"""


def names(action_numbers):
    """What each of the 4A + 1 numbers is, in order."""
    pushes = [f'number {number} {way}' for number in range(action_numbers) for way in ('up', 'down')]
    return (['cost, held'] + [f'cost change, {push}' for push in pushes]
            + [f'reward change, {push}' for push in pushes])


def pushed(proposed, push=1.0):
    """The 2A + 1 actions the look-ahead holds, for every state of a batch.

    proposed    the action the frozen agent proposed in each state
    push        how far a number is pushed towards its limit: 1 is all the way, 0.5 half way

    Returns one row per future and one column per state. Row 0 is the proposed action; then
    number 0 pushed up, number 0 pushed down, and so on.
    """
    import jax.numpy as jnp

    action_numbers = proposed.shape[1]
    # The simulator clips an action to its limits, so the future is imagined with the clipped one.
    held = jnp.clip(proposed.astype(jnp.float32), -1, 1)
    actions = [held]
    for number in range(action_numbers):
        for limit in (1.0, -1.0):
            moved = held[:, number] + push * (limit - held[:, number])
            actions.append(jnp.concatenate([held[:, :number], moved[:, None], held[:, number + 1:]], -1))
    return jnp.stack(actions)


def hold(frozen, predictor, state, actions, decisions):
    """Imagines holding each of `actions` for `decisions` decisions, from every state of a batch.

    It is written with the pieces of frozen_agent.py, as is everything else here that takes
    `frozen`, so it has to be called from a function that was passed to FrozenDreamer.compiled.

    predictor   the cost predictor's weights, offset and mean_cost
    state       the internal states, one row each
    actions     the actions to hold, one row per future and one column per state
    decisions   how many decisions each future lasts

    Returns three arrays with one row per future and one column per state:

        cost     the expected cost, added up over the decisions
        reward   the predicted reward, added up over the decisions
        value    the value estimate of the last state
    """
    import jax.numpy as jnp
    from cost_predictor import expected_cost

    count, states, action_numbers = actions.shape
    # All the futures of all the states are rolled together, as one batch: first every
    # state with its first future, then every state with its second, and so on.
    actions = actions.reshape(count * states, action_numbers)
    state = {name: jnp.concatenate([part] * count) for name, part in state.items()}
    cost = reward = 0.0
    for _ in range(decisions):
        state = frozen.roll_forward(state, actions, sample=False)
        cost = cost + expected_cost(predictor, frozen.numbers(state))
        reward = reward + frozen.reward(state).astype(jnp.float32)
    value = frozen.value(state).astype(jnp.float32)
    return {'cost': cost.reshape(count, states), 'reward': reward.reshape(count, states),
            'value': value.reshape(count, states)}


def futures(frozen, predictor, state, proposed, decisions, push=1.0):
    """Imagines the 2A + 1 futures of every state of a batch.

    Returns what pushed() gives, as `action`, with what hold() gives for those actions.
    """
    actions = pushed(proposed, push)
    return {'action': actions, **hold(frozen, predictor, state, actions, decisions)}


def look_ahead(frozen, predictor, state, proposed, decisions, push=1.0):
    """The 4A + 1 numbers an adapter reads, for every state of a batch. The arguments are those of pushed() and hold()."""
    return numbers_of(futures(frozen, predictor, state, proposed, decisions, push))


def numbers_of(imagined):
    """The 4A + 1 numbers, from the futures that futures() imagined."""
    import jax.numpy as jnp
    cost = imagined['cost']
    # The value estimate stands for the reward still to come after the last imagined decision.
    reward = imagined['reward'] + imagined['value']
    return jnp.concatenate([cost[:1], cost[1:] - cost[:1], reward[1:] - reward[:1]]).T


def sizes(inputs):
    """The typical size of each look-ahead number, measured over many decisions. An adapter divides by it.

    inputs    the numbers of look_ahead(), one row per decision

    Three sizes are measured, each a root mean square: of the expected cost of holding, of
    the cost changes, and of the reward changes. All the pushes share one size, so a push
    that the model expects to matter twice as much as another still reads twice as large.
    Nothing is subtracted, so a number that is zero stays zero.
    """
    import numpy as np

    action_numbers = (inputs.shape[1] - 1) // 4
    measured = np.empty(inputs.shape[1], np.float32)
    for group in (slice(0, 1), slice(1, 1 + 2 * action_numbers), slice(1 + 2 * action_numbers, None)):
        measured[group] = np.sqrt(np.mean(np.square(inputs[:, group], dtype=np.float64)))
    if not (measured > 0).all():
        raise SystemExit('a group of look-ahead numbers is zero at every decision, so it has no size to divide by')
    return measured


# The look-ahead of the adapter with memory (implementation/adapterArchitecture.md, decision 8).
#
# It differs from the one above in three ways. It holds one action more, the proposed
# action scaled to nothing, so that a gain has an input of its own. It is read over
# several stretches of one roll (decisions 1 to 5, 6 to 10 and 11 to 15) and not at one
# length. And its numbers are squashed by a symmetric log after the division by sizes.
#
# These functions are written with frozen.xp and frozen.step, which both frozen agents
# have, so the same lines run on DreamerV3 (jax) and on TD-MPC2 (torch).

LENGTHS = (5, 10, 15)    # where the stretches end, in decisions


def names_by_stretch(action_numbers, lengths=LENGTHS, stop=True):
    """What each number of numbers_by_stretch() is, in order."""
    others = [f'number {number} {way}' for number in range(action_numbers) for way in ('up', 'down')]
    others += ['stopped'] if stop else []
    called, first = [], 1
    for last in lengths:
        stretch = f'decisions {first} to {last}'
        called += ([f'cost, held, {stretch}'] + [f'cost change, {other}, {stretch}' for other in others]
                   + [f'reward change, {other}, {stretch}' for other in others])
        first = last + 1
    return called


def held(frozen, proposed, stop=True):
    """The actions the look-ahead holds, for every state of a batch: one row per future, one column per state.

    Row 0 is the proposed action. Then each action number in turn is set to +1 and to -1.
    With `stop`, the last row is the proposed action scaled to nothing.
    """
    xp = frozen.xp
    # The simulator clips an action to its limits, so the future is imagined with the clipped one.
    base = xp.clip(frozen.as_float(proposed), -1, 1)
    actions = [base]
    for number in range(base.shape[1]):
        for limit in (1.0, -1.0):
            moved = base[:, number:number + 1] * 0 + limit
            actions.append(xp.concatenate([base[:, :number], moved, base[:, number + 1:]], -1))
    if stop:
        actions.append(base * 0)
    return xp.stack(actions)


def expected(xp, predictor, numbers):
    """The expected cost of arriving in each state: the chance of a cost times the mean cost of a costly decision.

    `predictor` holds weights, offset and mean_cost. If it also holds `first` and
    `first_offset`, the state numbers pass through one hidden layer on the way.
    """
    if 'first' in predictor:
        numbers = xp.tanh(numbers @ predictor['first'] + predictor['first_offset'])
    score = numbers @ predictor['weights'] + predictor['offset']
    return predictor['mean_cost'] / (1 + xp.exp(-score))


def hold_by_stretch(frozen, predictor, state, actions, lengths=LENGTHS):
    """Imagines holding each of `actions` to the last of `lengths`, and reads the sums at every one of them.

    Returns `cost` and `reward`, each with one layer per length, one row per future and
    one column per state. A cost is the expected cost added up to that length. A reward
    is the predicted reward added up to that length, with the value estimate of the state
    reached there for the reward still to come.
    """
    xp = frozen.xp
    count, states, action_numbers = actions.shape
    # All the futures of all the states are rolled together, as one batch.
    actions = actions.reshape(count * states, action_numbers)
    state = {name: xp.concatenate([state[name]] * count) for name in frozen.names}
    cost = reward = 0.0
    costs, rewards = [], []
    for decision in range(1, max(lengths) + 1):
        state, earned = frozen.step(state, actions, sample=False)
        cost = cost + expected(xp, predictor, frozen.numbers(state))
        reward = reward + earned
        if decision in lengths:
            costs.append(cost.reshape(count, states))
            rewards.append((reward + frozen.worth(state)).reshape(count, states))
    return {'cost': xp.stack(costs), 'reward': xp.stack(rewards)}


def numbers_by_stretch(frozen, imagined):
    """The look-ahead numbers of every state, one row each, from what hold_by_stretch() imagined.

    For each length: the cost of holding the proposed action, then how much each other
    held action changes the cost, then how much each changes the reward. The first stretch
    is that at the first length, and every later stretch is what going on to the next
    length adds. names_by_stretch() gives the order.
    """
    xp = frozen.xp
    cost, reward = imagined['cost'], imagined['reward']
    stretches, before = [], None
    for length in range(cost.shape[0]):
        up_to = xp.concatenate([cost[length, :1], cost[length, 1:] - cost[length, :1],
                                reward[length, 1:] - reward[length, :1]])
        stretches.append(up_to if before is None else up_to - before)
        before = up_to
    return xp.concatenate(stretches).T


def sizes_by_stretch(inputs, stretches=len(LENGTHS)):
    """The typical size of each number of numbers_by_stretch(), measured over many decisions.

    Within each stretch three sizes are measured, each a root mean square: of the cost of
    holding, of the cost changes, and of the reward changes. Nothing is subtracted, so a
    number that is zero stays zero.
    """
    import numpy as np

    each = inputs.shape[1] // stretches
    others = (each - 1) // 2
    measured = np.empty(inputs.shape[1], np.float32)
    for stretch in range(stretches):
        first = stretch * each
        for group in (slice(first, first + 1), slice(first + 1, first + 1 + others), slice(first + 1 + others, first + each)):
            measured[group] = np.sqrt(np.mean(np.square(inputs[:, group], dtype=np.float64)))
    if not (measured > 0).all():
        raise SystemExit('a group of look-ahead numbers is zero at every decision, so it has no size to divide by')
    return measured


def squashed(xp, numbers, sizes):
    """The look-ahead numbers as an adapter with memory reads them: divided by their sizes, then a symmetric log.

    The symmetric log keeps the sign and keeps zero at zero, and brings the large values
    near danger within reach of the small ones away from it.
    """
    scaled = numbers / sizes
    return xp.sign(scaled) * xp.log1p(xp.abs(scaled))


def read_by_stretch(frozen, predictor, state, proposed, lengths=LENGTHS, stop=True):
    """The unsquashed look-ahead numbers of every state of a batch, by stretch."""
    return numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, held(frozen, proposed, stop), lengths))


def measured_by_stretch(frozen, predictor, files, lengths=LENGTHS, stop=True):
    """The sizes of the look-ahead numbers over every decision of the saved episodes in `files`."""
    import numpy as np

    def look(state, proposed, predictor):
        return read_by_stretch(frozen, predictor, state, proposed, lengths, stop)

    look = frozen.compiled(look)
    numbers = []
    for file in files:
        saved = np.load(file)
        numbers.append(look(frozen.written_out(frozen.kept(saved)), saved['proposed'], predictor))
    return sizes_by_stretch(np.concatenate(numbers), len(lengths))
