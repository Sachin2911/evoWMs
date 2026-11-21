"""The adapter: a small correction to the frozen agent's action, read off the world model's state.

At every decision the frozen actor proposes an action. The adapter looks at the world
model's 512 continuous numbers ("deter") and changes that action:

    action = proposed * (1 + gain) + shift

where gain and shift are weighted sums of what the adapter reads. All weights start at zero,
which leaves the action exactly as proposed: nothing about what "safe" means is built in.
The weights are one flat list of numbers, which is what evolution searches over.

What the adapter reads is one of:

  hidden  the 16 hidden units of the small cost network (cost_features.py)
  deter   the 512 numbers themselves, each scaled to a similar range
  gated   the 16 hidden units again, but gain and shift are multiplied by the cost network's
          chance of a cost within the next 5 decisions. The adapter then has little say while
          no cost is in sight, and cannot make the agent safe by simply holding it still.

Every function takes `xp`, which is numpy when playing in the simulator and jax.numpy when
scoring candidates inside the world model, so both use the same arithmetic.
"""

import numpy as np

KINDS = ('hidden', 'deter', 'gated')


def count_features(net, kind):
    return net['w1'].shape[0] if kind == 'deter' else net['w1'].shape[1]


def count_weights(net, kind, action_size):
    # For each action dimension a gain and a shift, each a weighted sum of the features plus a constant.
    return (count_features(net, kind) + 1) * 2 * action_size


def scaled(net, deter, xp):
    """The 512 numbers with the training episodes' mean taken off and divided by their spread."""
    return (deter.astype(xp.float32) - net['mean']) / net['spread']


def hidden(net, deter, xp):
    return xp.tanh(scaled(net, deter, xp) @ net['w1'] + net['b1'])


def cost_logits(net, deter, xp):
    """The cost network's outputs, one column per name in cost_features.OUTPUTS."""
    return hidden(net, deter, xp) @ net['w2'] + net['b2']


def adapt(weights, net, kind, deter, action, xp):
    """The adapter's action. weights: flat; deter: (..., 512); action: (..., action_size)."""
    feats = scaled(net, deter, xp) if kind == 'deter' else hidden(net, deter, xp)
    count, size = feats.shape[-1], action.shape[-1]
    # Divided by the square root of the number of features, so a random change to the weights
    # moves the action by a similar amount whether the adapter reads 16 numbers or 512.
    matrix = weights[:count * 2 * size].reshape(count, 2 * size) / np.sqrt(count)
    out = feats @ matrix + weights[count * 2 * size:]
    if kind == 'gated':
        # Column 1 of the cost network answers "a cost within the next 5 decisions?".
        ahead = feats @ net['w2'][:, 1] + net['b2'][1]
        out = out * (1 / (1 + xp.exp(-ahead)))[..., None]
    return action * (1 + out[..., :size]) + out[..., size:]


def adapted_policy(agent, net, kind, get_weights, on_decision=None):
    """The frozen agent's policy with the adapter applied, for SafeDreamer's episode driver.

    get_weights() returns the weights to use now, so the caller can swap candidates between
    episodes. on_decision(latent) is called with the world model's state at every decision.
    """
    def policy(*inputs):
        actions, state = agent.policy(*inputs, mode='eval')
        (latent, action), task_state, exploration_state = state
        if on_decision:
            on_decision(latent)
        adapted = adapt(get_weights(), net, kind, latent['deter'], action, np).astype(action.dtype)
        # The world model is told the action that was really taken, not the one first proposed.
        return {**actions, 'action': adapted}, ((latent, adapted), task_state, exploration_state)
    return policy


def save(path, weights, kind, net, **notes):
    """One file with everything needed to run the adapter: its weights and the cost network it reads through."""
    np.savez(path, weights=np.asarray(weights, np.float32), kind=kind,
             **{f'net_{name}': value for name, value in net.items()}, **notes)


def load(path):
    data = np.load(path)
    net = {name[len('net_'):]: data[name] for name in data.files if name.startswith('net_')}
    return data['weights'], str(data['kind']), net
