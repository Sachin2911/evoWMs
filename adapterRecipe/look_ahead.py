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

names() gives their order, and futures() gives the imagined futures themselves.

Why it is built this way:

  The action is held, and the agent is not asked again on the way. That needs only the
  model, the reward predictor, the value estimate and the cost predictor. Asking the agent
  for an action in every imagined state would be cheap for DreamerV3, but TD-MPC2 proposes
  by planning, which is far too slow to do inside every imagined step.

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


def futures(frozen, predictor, state, proposed, decisions, push=1.0):
    """Imagines the 2A + 1 futures of every state of a batch.

    Like look_ahead() below, it is written with the pieces of frozen_agent.py, so it has to
    be called from a function that was passed to FrozenDreamer.compiled.

    predictor   the cost predictor's weights, offset and mean_cost
    state       the internal states, one row each
    proposed    the action the frozen agent proposed in each
    decisions   how many decisions each future lasts
    push        how far a number is pushed towards its limit: 1 is all the way, 0.5 half way

    Returns four arrays with one row per future and one column per state. The proposed
    action's own future is row 0; then number 0 pushed up, number 0 pushed down, and so on.

        action   the action that is held
        cost     the expected cost, added up over the decisions
        reward   the predicted reward, added up over the decisions
        value    the value estimate of the last state
    """
    import jax.numpy as jnp
    from cost_predictor import expected_cost

    states, action_numbers = proposed.shape
    # The simulator clips an action to its limits, so the future is imagined with the clipped one.
    held = jnp.clip(proposed.astype(jnp.float32), -1, 1)
    actions = [held]
    for number in range(action_numbers):
        for limit in (1.0, -1.0):
            pushed = held[:, number] + push * (limit - held[:, number])
            actions.append(jnp.concatenate([held[:, :number], pushed[:, None], held[:, number + 1:]], -1))

    # All the futures of all the states are rolled together, as one batch: first every
    # state with its first future, then every state with its second, and so on.
    count = len(actions)
    actions = jnp.concatenate(actions)
    state = {name: jnp.concatenate([part] * count) for name, part in state.items()}
    cost = reward = 0.0
    for _ in range(decisions):
        state = frozen.roll_forward(state, actions, sample=False)
        cost = cost + expected_cost(predictor, frozen.numbers(state))
        reward = reward + frozen.reward(state).astype(jnp.float32)
    value = frozen.value(state).astype(jnp.float32)
    return {'action': actions.reshape(count, states, action_numbers), 'cost': cost.reshape(count, states),
            'reward': reward.reshape(count, states), 'value': value.reshape(count, states)}


def look_ahead(frozen, predictor, state, proposed, decisions, push=1.0):
    """The 4A + 1 numbers an adapter reads, for every state of a batch. The arguments are those of futures()."""
    import jax.numpy as jnp
    imagined = futures(frozen, predictor, state, proposed, decisions, push)
    cost = imagined['cost']
    # The value estimate stands for the reward still to come after the last imagined decision.
    reward = imagined['reward'] + imagined['value']
    return jnp.concatenate([cost[:1], cost[1:] - cost[:1], reward[1:] - reward[:1]]).T
