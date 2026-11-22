"""The adapter of the recipe search: a correction to the frozen agent's action that is only
switched on when a cost is predicted.

At every decision the frozen actor proposes an action. The adapter changes it:

    action = proposed * (1 + gate * gain) + gate * shift

gain and shift are weighted sums of what the adapter reads, plus a constant. All weights
start at zero, which leaves the action exactly as proposed. What to do when a cost is
predicted (brake, reverse, turn, and which way) is not built in: evolution has to find it.

The gate (--gate) is a number between 0 and 1 from the small cost network that was trained
on the unadapted agent's first episodes (adapterPilot/cost_features.py):

  none      always 1: the pilot's adapter, which acts everywhere
  arrived   the chance that the decision that led here cost
  ahead5    the chance of a cost within the next 5 decisions
  ahead15   ... within the next 15

With a gate the adapter cannot change what the agent does where no cost is predicted. In
particular it cannot make the agent stand still everywhere, which is the easy way to be
safe that the pilot's adapters all found.

What the adapter reads (--reads), all from the world model's 512 continuous numbers:

  const   nothing: gain and shift are constants (4 weights for a 2-number action)
  probs   the cost network's three chances
  hidden  its 16 hidden units and its three chances
  deter   the 512 numbers themselves, scaled, and the three chances
  look    what the frozen world model expects of a few alternatives to the proposed action,
          and the three chances. See look_ahead below.
  look+hidden   both

Every function takes `xp`, which is numpy when playing in the simulator and jax.numpy when
scoring candidates inside the world model, so both use the same arithmetic.
"""

import numpy as np

READS = ('const', 'probs', 'hidden', 'deter', 'look', 'look+hidden')
LOOKS = ('look', 'look+hidden')                 # the kinds that need look_ahead
DANGER_SCALE, REWARD_SCALE = 100.0, 20.0        # bring the look-ahead numbers to a size of about 1
# A gate named like ahead15-full10 is fully open once that chance reaches 10%, and closes in proportion below it.
GATES = ('none', 'arrived', 'ahead5', 'ahead15', 'ahead15-full05', 'ahead15-full10', 'ahead15-full20')
OUTPUTS = ('arrived', 'ahead5', 'ahead15')      # the cost network's outputs, in this order


def scaled(net, deter, xp):
    """The 512 numbers with the training episodes' mean taken off and divided by their spread."""
    return (deter.astype(xp.float32) - net['mean']) / net['spread']


def hidden(net, deter, xp):
    return xp.tanh(scaled(net, deter, xp) @ net['w1'] + net['b1'])


def cost_logits(net, deter, xp):
    """The cost network's outputs before they are squashed into chances."""
    return hidden(net, deter, xp) @ net['w2'] + net['b2']


def count_features(net, reads, action_size):
    looked = 4 * action_size + 1
    return {'const': 0, 'probs': len(OUTPUTS), 'hidden': net['w1'].shape[1] + len(OUTPUTS),
            'deter': net['w1'].shape[0] + len(OUTPUTS), 'look': looked + len(OUTPUTS),
            'look+hidden': looked + net['w1'].shape[1] + len(OUTPUTS)}[reads]


def count_weights(net, reads, action_size):
    # For each action dimension a gain and a shift, each a weighted sum of the features plus a constant.
    return (count_features(net, reads, action_size) + 1) * 2 * action_size


def look_ahead(world, state, action, net, horizon, samples, xp, push=1.0, scan=None):
    """What the frozen world model expects if the proposed action, or a variation of it, is held for a while.

    The variations push one action dimension at a time up and down by `push` (kept within -1
    to 1); with 2 action dimensions that is 5 actions to try. Each is held for `horizon`
    decisions inside the world model, `samples` times over. The cost network then gives the
    chance of a cost along the way and the world model's reward predictor the reward.

    Returns, for every state: the proposed action's chance of a cost, how much each variation
    changes that chance, and how much each variation changes the reward. Nothing here says
    which variation to prefer; the adapter's weights, found by evolution, decide that.

    Only runs inside JAX (xp is jax.numpy): state is the world model's state with 'deter'
    (..., 512) and 'stoch', action is (batch, action_size). scan is SafeDreamer's ninjax.scan;
    with it the roll-out is one compiled loop and not `horizon` copies of the world model.
    """
    batch, size = action.shape
    pushes = xp.concatenate([xp.zeros((1, size)), xp.eye(size), -xp.eye(size)], 0) * push
    tried = xp.clip(action.astype(xp.float32)[:, None, :] + pushes[None], -1, 1)        # (batch, 1 + 2 size, size)
    count = pushes.shape[0]
    tried = xp.repeat(tried[:, :, None, :], samples, 2).reshape(batch * count * samples, size)
    current = {name: xp.repeat(state[name], count * samples, 0) for name in ('deter', 'stoch')}

    def held(current, _):
        following = world.rssm.img_step(current, tried)
        chance = 1 / (1 + xp.exp(-cost_logits(net, following['deter'], xp)[..., 0]))
        reward = world.heads['reward'](following).mean().astype(xp.float32)
        return {name: following[name] for name in ('deter', 'stoch')}, (chance, reward)

    # The first decision is taken outside the loop: its result has exactly the form the loop must keep.
    current, (chance, reward) = held(current, 0)
    if horizon > 1 and scan:
        _, (chances, rewards) = scan(held, current, xp.arange(horizon - 1))
        chance, reward = (chance + chances.sum(0)) / horizon, (reward + rewards.sum(0)) / horizon
    elif horizon > 1:
        for _ in range(horizon - 1):
            current, (more_chance, more_reward) = held(current, 0)
            chance, reward = chance + more_chance, reward + more_reward
        chance, reward = chance / horizon, reward / horizon
    chance = chance.reshape(batch, count, samples).mean(-1) * DANGER_SCALE
    reward = reward.reshape(batch, count, samples).mean(-1) * REWARD_SCALE
    return xp.concatenate([chance[:, :1], chance[:, 1:] - chance[:, :1], reward[:, 1:] - reward[:, :1]], -1)


def read(net, deter, reads, gate, xp, looked=None):
    """What the adapter reads at these states, and the gate's value there.

    looked is look_ahead's answer for these states, needed when reads is one of LOOKS.
    """
    units = hidden(net, deter, xp)
    chances = 1 / (1 + xp.exp(-(units @ net['w2'] + net['b2'])))
    if reads == 'const':
        feats = chances[..., :0]
    elif reads == 'probs':
        feats = chances
    elif reads == 'hidden':
        feats = xp.concatenate([units, chances], -1)
    elif reads == 'look':
        feats = xp.concatenate([looked, chances], -1)
    elif reads == 'look+hidden':
        feats = xp.concatenate([looked, units, chances], -1)
    else:
        feats = xp.concatenate([scaled(net, deter, xp), chances], -1)
    if gate == 'none':
        opening = 1.0
    else:
        name, _, full = gate.partition('-full')
        opening = chances[..., OUTPUTS.index(name)][..., None]
        if full:
            # The plain gate is small exactly where steering early would still help: at a 10% chance
            # the adapter's push is a tenth of what it asks for. This one gives it its full say there,
            # and still shuts it where the cost network sees no danger.
            opening = xp.minimum(1.0, opening / (int(full) / 100))
    return feats, opening


def adapt(weights, feats, opening, action, xp, bound=0.0):
    """The adapter's action. weights: flat; feats: (..., features); action: (..., action_size).

    With a bound, gain and shift are squashed to stay between -bound and bound before the gate
    is applied. The adapter can then only cancel the proposed action outright where the gate
    is at least 1 / bound; without it, large weights make the gate meaningless.
    """
    count, size = feats.shape[-1], action.shape[-1]
    out = weights[count * 2 * size:]
    if count:
        # Divided by the square root of the number of features, so a random change to the weights
        # moves the action by a similar amount whether the adapter reads 3 numbers or 515.
        out = feats @ (weights[:count * 2 * size].reshape(count, 2 * size) / np.sqrt(count)) + out
    if bound:
        out = bound * xp.tanh(out / bound)
    out = out * opening
    # The simulator only accepts actions between -1 and 1; the world model is given the same.
    return xp.clip(action * (1 + out[..., :size]) + out[..., size:], -1, 1)


def adapted_policy(agent, net, reads, gate, get_weights, on_decision=None, bound=0.0, look=None):
    """The frozen agent's policy with the adapter applied, for SafeDreamer's episode driver.

    get_weights() returns the weights to use now, so the caller can swap candidates between
    episodes. on_decision(latent) is called with the world model's state at every decision.
    look(latent, action) gives look_ahead's answer as a numpy array, for the kinds in LOOKS.
    """
    def policy(*inputs):
        actions, state = agent.policy(*inputs, mode='eval')
        (latent, action), task_state, exploration_state = state
        if on_decision:
            on_decision(latent)
        looked = look(latent, action) if reads in LOOKS else None
        feats, opening = read(net, latent['deter'], reads, gate, np, looked)
        adapted = adapt(get_weights(), feats, opening, action, np, bound).astype(action.dtype)
        # The world model is told the action that was really taken, not the one first proposed.
        return {**actions, 'action': adapted}, ((latent, adapted), task_state, exploration_state)
    return policy


def save(path, weights, reads, gate, net, bound=0.0, **notes):
    """One file with everything needed to run the adapter: its weights and the cost network it reads through."""
    np.savez(path, weights=np.asarray(weights, np.float32), reads=reads, gate=gate, bound=np.float32(bound),
             **{f'net_{name}': value for name, value in net.items()}, **notes)


def load(path):
    """Returns weights, reads, gate, the cost network, and the bound on gain and shift (0 for none)."""
    data = np.load(path)
    net = {name[len('net_'):]: data[name] for name in data.files if name.startswith('net_')}
    bound = float(data['bound']) if 'bound' in data.files else 0.0
    return data['weights'], str(data['reads']), str(data['gate']), net, bound
