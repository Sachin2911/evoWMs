"""The frozen DreamerV3 agent, as the five pieces the adapter recipe is allowed to use.

    state                         the world model's internal state after the latest observation
    roll_forward(state, action)   the state the world model expects after `action`
    reward(state)                 the reward it predicts for arriving in `state`
    value(state)                  its estimate of the reward still to come from `state`
    propose(state)                the action the agent's actor takes in `state`

The recipe reads the agent through these and nothing else, so that it can be put on
TD-MPC2 by writing the same five for that agent (brainStorming/adapterRecipeDesign.md,
decision 1). frozen_tdmpc2.py does that, and what the two share is this:

    xp                            the array functions the pieces work with (jax.numpy here)
    step(state, action)           roll_forward and reward in one: the state that follows,
                                  and the reward predicted for the decision
    value(state)                  as above, as ordinary numbers
    propose(state)                as above
    numbers(state)                the state as one row of numbers
    compiled(function)            `function`, written with the pieces, for plain arrays
    kept(episode)                 the states of a saved episode, in the small form they are
                                  saved in, and written_out() to hand them to the pieces

Nothing here changes the agent. Its weights are read and never written, and when it plays
it is called exactly as safeDreamerBaselines/evaluate.py calls it. An adapter is applied
afterwards, to the action that comes out.
"""

import numpy as np


class FrozenDreamer:
    """Wraps a loaded reward-only SafeDreamer agent (the `dreamerv3` variant)."""

    def __init__(self, agent):
        # `agent` is SafeDreamer's outer object. The networks are one level in, at agent.agent.
        inner = agent.agent
        if inner.config.use_cost:
            raise ValueError('FrozenDreamer is for the reward-only agent, which has no cost networks')
        self._agent = agent
        self._model = inner.wm.rssm
        self._reward = inner.wm.heads['reward']
        self._actor = inner.task_behavior.ac.actor
        self._critic = inner.task_behavior.ac.critics['extr'].net
        self._device = agent.policy_devices[0]
        self._random = np.random.default_rng(0)

    # The pieces. They run on the GPU inside SafeDreamer's own machinery, so they can only be
    # called from a function that has been passed to compiled() below. Each takes and
    # returns a batch: one row per state.

    def roll_forward(self, state, action, sample=True):
        """The state the world model expects after taking `action` in `state`.

        Part of the model's next state is a random draw, so two calls can differ. With
        sample=False each categorical choice is instead the one the model finds most
        likely, and the same state and action always give the same next state.
        """
        import jax
        import jax.numpy as jnp
        following = self._model.img_step(state, action)
        if sample:
            return following
        # `logit` holds how likely the model finds each class of each choice.
        likely = jnp.argmax(following['logit'], -1)
        classes = following['logit'].shape[-1]
        return {**following, 'stoch': jax.nn.one_hot(likely, classes, dtype=following['stoch'].dtype)}

    def reward(self, state):
        """The reward the world model predicts for the decision that led to `state`."""
        return self._reward(state).mean()

    @property
    def xp(self):
        """The array functions the pieces work with. The adapter with memory is written with these."""
        import jax.numpy as jnp
        return jnp

    def as_float(self, array):
        """`array` as ordinary numbers. The agent itself computes in half the digits."""
        import jax.numpy as jnp
        return array.astype(jnp.float32)

    def step(self, state, action, sample=True):
        """One decision in the world model: the state that follows `action`, and the reward predicted for it."""
        following = self.roll_forward(state, action, sample)
        following = {name: following[name] for name in ('deter', 'stoch')}
        return following, self.as_float(self.reward(following))

    def worth(self, state):
        """value(), as ordinary numbers."""
        return self.as_float(self.value(state))

    # A saved episode keeps each categorical choice as its number. These turn the states
    # of an episode into that small form, and back into what the pieces take.

    names = ('deter', 'stoch')

    def kept(self, episode, entries=slice(None)):
        """The states at the chosen entries of a saved episode, in the small form they are saved in."""
        return {'deter': episode['deter'][entries], 'stoch': episode['stoch'][entries]}

    def written_out(self, kept):
        """States in the small form, as the pieces take them: every choice written out as zeros and a one."""
        return {'deter': kept['deter'], 'stoch': np.eye(self._model._classes, dtype=np.float16)[kept['stoch']]}

    def rows(self, kept):
        """States in the small form, as the rows of numbers the cost predictor is fitted to."""
        choices = np.eye(self._model._classes, dtype=np.float32)[kept['stoch']]
        return np.concatenate([kept['deter'].astype(np.float32), choices.reshape(len(choices), -1)], -1)

    def value(self, state):
        """The agent's own estimate of the discounted reward still to come from `state`."""
        return self._critic(state).mean()

    def propose(self, state, sample=True):
        """The action the frozen actor takes in `state`.

        A random draw from the actor's choice, as when the agent plays, or with sample=False
        its most likely action.
        """
        from SafeDreamer import ninjax
        choice = self._actor(state)
        return choice.sample(seed=ninjax.rng()) if sample else choice.mode()

    def numbers(self, state):
        """The state as one row of numbers, the way the agent's own networks read it.

        512 continuous numbers, then the 48 categorical choices written out as 48 x 48 zeros
        and ones: 2,816 numbers in all.
        """
        import jax.numpy as jnp
        choices = state['stoch']
        choices = choices.reshape(choices.shape[:-2] + (-1,))
        return jnp.concatenate([state['deter'], choices], -1).astype(jnp.float32)

    def compiled(self, function):
        """Makes `function`, written with the pieces above, callable with plain arrays.

        The result runs on the GPU with the frozen weights. It takes numpy arrays, or tuples
        and dicts of them, and returns numpy arrays.

        It stops if `function` would use a network that did not come with the checkpoint.
        SafeDreamer would otherwise create that network with random weights and carry on
        without saying so.
        """
        import jax
        from SafeDreamer import ninjax

        run = ninjax.jit(ninjax.pure(function), device=self._device)

        def call(*inputs):
            weights = self._agent.varibs
            # SafeDreamer forbids silent copies between the computer's memory and the GPU's,
            # so every move in or out is spelt out here.
            seed = jax.device_put(self._random.integers(2 ** 63 - 1), self._device)
            inputs = jax.device_put(inputs, self._device)
            output, after = run(weights, seed, *inputs)
            created = sorted(after.keys() - weights.keys())
            if created:
                raise RuntimeError(
                    f'{function.__name__} would use {len(created)} networks that are not in '
                    f'the checkpoint, e.g. {created[:3]}')
            return jax.tree_util.tree_map(np.asarray, jax.device_get(output))

        return call

    def reseed(self, seed):
        """Fixes the random numbers used by everything made with compiled().

        They are kept apart from the agent's own random numbers (agent.rng), so that nothing
        an adapter computes can change what the frozen agent would have done.
        """
        # [seed, 1] gives a different sequence from default_rng(seed), which is what
        # evaluate.py gives the agent for the same episode.
        self._random = np.random.default_rng([seed, 1])

    def policy(self, observation, carry, adapter=None):
        """One decision in the simulator, in the form SafeDreamer's episode loop expects.

        The frozen agent decides first, called exactly as when it plays alone. `adapter`, if
        given, is then called as adapter(state, proposed), with the agent's internal state
        and the action it proposed, and returns the action to take. Both are numpy arrays
        with one row, because the loop plays one simulator.
        """
        outputs, carry = self._agent.policy(observation, carry, mode='eval')
        if adapter is None:
            return outputs, carry
        # What the agent carries to its next decision: its state and the action it believes
        # it took, then two entries that are empty for this agent.
        (state, proposed), *rest = carry
        action = np.asarray(adapter(state, proposed), proposed.dtype)
        if action.shape != proposed.shape:
            raise ValueError(f'the adapter returned shape {action.shape}, not {proposed.shape}')
        # The world model has to be told the action that was really taken. Otherwise its
        # picture of the world drifts away from the simulator.
        return {**outputs, 'action': action}, ((state, action), *rest)
