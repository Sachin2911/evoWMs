"""Tests of the adapter with memory that need no agent, no simulator and no GPU.

The adapter, its look-ahead and the imagined rolls are written with the frozen agent's
own array functions. Here a made-up agent stands in, whose pieces are a few lines of
numpy, so the arithmetic can be checked on any computer:

    python3 -m unittest test_memory

The checks with the real agent are in check_memory.py, which runs on the cluster.
"""

import pathlib
import tempfile
import unittest

import numpy as np

import adapter as adapters
from imagine import Imagination
from look_ahead import (LENGTHS, held, hold_by_stretch, names_by_stretch, numbers_by_stretch, sizes_by_stretch,
                        squashed)


class MadeUpAgent:
    """Five pieces in numpy: a state of S numbers, a model that mixes state and action, and sums for reward and value."""

    xp = np
    names = ('z',)

    def __init__(self, state_numbers=6, action_numbers=2, seed=0):
        random = np.random.default_rng(seed)
        self._from_state = random.normal(size=(state_numbers, state_numbers)).astype(np.float32) / 2
        self._from_action = random.normal(size=(action_numbers, state_numbers)).astype(np.float32)
        self._to_action = random.normal(size=(state_numbers, action_numbers)).astype(np.float32)
        self._random = np.random.default_rng(seed)

    def as_float(self, array):
        return array.astype(np.float32)

    def step(self, state, action, sample=True):
        # Like both real agents, the model clips the action it is given.
        following = np.tanh(state['z'] @ self._from_state + np.clip(action, -1, 1) @ self._from_action)
        if sample:
            following = following + 0.01 * self._random.normal(size=following.shape).astype(np.float32)
        return {'z': following}, following.sum(-1)

    def worth(self, state):
        return state['z'][:, 0] * 2

    def propose(self, state, sample=True):
        # Some proposals lie beyond the limits, as the real agents' can.
        return 1.5 * np.tanh(state['z'] @ self._to_action)

    def numbers(self, state):
        return state['z']

    def compiled(self, function):
        return function

    def reseed(self, seed):
        self._random = np.random.default_rng(seed)


def made_up(states=7, state_numbers=6, action_numbers=2):
    frozen = MadeUpAgent(state_numbers, action_numbers)
    random = np.random.default_rng(1)
    state = {'z': random.normal(size=(states, state_numbers)).astype(np.float32)}
    predictor = {'weights': random.normal(size=state_numbers).astype(np.float32), 'offset': np.float32(-1),
                 'mean_cost': np.float32(1.5)}
    return frozen, state, predictor


class LookAhead(unittest.TestCase):

    def test_held_actions(self):
        frozen, state, _ = made_up()
        proposed = frozen.propose(state)
        actions = held(frozen, proposed)
        self.assertEqual(actions.shape, (6, 7, 2))
        np.testing.assert_array_equal(actions[0], np.clip(proposed, -1, 1))
        np.testing.assert_array_equal(actions[1][:, 0], 1)        # number 0 up
        np.testing.assert_array_equal(actions[1][:, 1], actions[0][:, 1])
        np.testing.assert_array_equal(actions[4][:, 1], -1)       # number 1 down
        np.testing.assert_array_equal(actions[5], 0)              # stopped
        self.assertEqual(held(frozen, proposed, stop=False).shape, (5, 7, 2))

    def test_stretches_add_up_to_the_whole_roll(self):
        frozen, state, predictor = made_up()
        actions = held(frozen, frozen.propose(state))
        by_stretch = numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, actions))
        self.assertEqual(by_stretch.shape, (7, 33))
        self.assertEqual(len(names_by_stretch(2)), 33)
        whole = numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, actions, (15,)))
        np.testing.assert_allclose(by_stretch.reshape(7, 3, 11).sum(1), whole, rtol=1e-4, atol=1e-4)
        first = numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, actions, (5,)))
        np.testing.assert_allclose(by_stretch[:, :11], first, rtol=1e-5, atol=1e-5)

    def test_a_push_that_changes_nothing_reads_zero(self):
        frozen, state, predictor = made_up()
        proposed = frozen.propose(state)
        called = names_by_stretch(2)
        numbers = numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, held(frozen, proposed)))
        at_upper_limit = proposed[:, 0] >= 1
        self.assertTrue(at_upper_limit.any())
        for column, name in enumerate(called):
            if 'number 0 up' in name:
                np.testing.assert_array_equal(numbers[at_upper_limit, column], 0)
        # Where nothing is proposed, stopping changes nothing.
        nothing = numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, held(frozen, proposed * 0)))
        for column, name in enumerate(called):
            if 'stopped' in name:
                np.testing.assert_array_equal(nothing[:, column], 0)

    def test_sizes_and_squashing(self):
        frozen, state, predictor = made_up(states=50)
        numbers = numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, held(frozen, frozen.propose(state))))
        sizes = sizes_by_stretch(numbers)
        self.assertEqual(sizes.shape, (33,))
        # Within a stretch the cost changes share one size, and the reward changes another.
        self.assertEqual(len(set(sizes[1:6])), 1)
        self.assertEqual(len(set(sizes[6:11])), 1)
        self.assertEqual(len(set(sizes)), 9)
        squashed_numbers = squashed(np, numbers, sizes)
        np.testing.assert_array_equal(np.sign(squashed_numbers), np.sign(numbers))
        np.testing.assert_array_equal(squashed(np, numbers * 0, sizes), 0)


class Adapter(unittest.TestCase):

    def fixed(self, frozen, state, predictor, adapter):
        actions = held(frozen, frozen.propose(state), adapter.stop)
        numbers = numbers_by_stretch(frozen, hold_by_stretch(frozen, predictor, state, actions, adapter.lengths))
        return {'predictor': predictor, 'sizes': sizes_by_stretch(numbers, len(adapter.lengths))}

    def test_number_of_weights(self):
        self.assertEqual(adapters.MemoryAdapter(2).count, 152)
        self.assertEqual(adapters.MemoryAdapter(4).count, 528)
        self.assertEqual(adapters.MemoryAdapter(2, memory=False).count, 132)
        self.assertEqual(adapters.MemoryAdapter(2, added_on=True).count, 156)
        self.assertEqual(adapters.MemoryAdapter(2, stop=False).count, 4 * (27 + 4) + 4)
        self.assertEqual(len(adapters.MemoryAdapter(2).names()), 37)

    def test_zero_weights_hand_back_the_proposed_action(self):
        for settings in ({}, {'memory': False}, {'added_on': True}, {'stop': False}):
            frozen, state, predictor = made_up()
            adapter = adapters.MemoryAdapter(2, **settings)
            fixed = self.fixed(frozen, state, predictor, adapter)
            proposed = frozen.propose(state)
            self.assertTrue((np.abs(proposed) > 1).any())
            made = adapter.decide(frozen, fixed, np.zeros(adapter.count, np.float32), state, proposed, adapter.start(7))
            np.testing.assert_array_equal(made['action'], proposed)
            np.testing.assert_array_equal(made['memory'], 0)
            np.testing.assert_array_equal(made['output'], 0)

    def test_output_is_bounded_and_moves_the_action(self):
        frozen, state, predictor = made_up()
        adapter = adapters.MemoryAdapter(2)
        fixed = self.fixed(frozen, state, predictor, adapter)
        weights = (50 * np.random.default_rng(3).normal(size=adapter.count)).astype(np.float32)
        made = adapter.decide(frozen, fixed, weights, state, frozen.propose(state), adapter.start(7))
        self.assertTrue((np.abs(made['output']) <= 1).all())
        # An action number the adapter leaves at its limit is handed back as the agent gave it, beyond the limit.
        proposed = frozen.propose(state)
        left = made['action'] == proposed
        self.assertTrue((np.abs(made['action'][~left]) <= 1).all())
        self.assertTrue((np.abs(made['memory']) <= 1).all())
        self.assertTrue((made['action'] != frozen.propose(state)).any())

    def test_memory_fades_towards_the_output(self):
        frozen, state, predictor = made_up()
        adapter = adapters.MemoryAdapter(2, added_on=True)
        fixed = self.fixed(frozen, state, predictor, adapter)
        weights = np.zeros(adapter.count, np.float32)
        adapter.table(weights)[:, -1] = 0.7          # every output is tanh(0.7), whatever is read
        weights[-adapter.remembered:] = np.log(0.25 / 0.75)    # a fade of 0.25
        memory = adapter.start(7)
        for decision in range(1, 6):
            made = adapter.decide(frozen, fixed, weights, state, frozen.propose(state), memory)
            memory = made['memory']
            np.testing.assert_allclose(memory, np.tanh(0.7) * (1 - 0.75 ** decision), rtol=1e-5)

    def test_a_unit_with_a_weight_above_one_on_its_own_memory_latches(self):
        frozen, state, predictor = made_up(states=1)
        adapter = adapters.MemoryAdapter(2, added_on=True)
        fixed = self.fixed(frozen, state, predictor, adapter)
        weights = np.zeros(adapter.count, np.float32)
        table = adapter.table(weights)
        shift = adapter.action_numbers        # the line of the first shift
        table[shift, adapter.inputs + shift] = 3.0
        memory = adapter.start(1)
        outputs = []
        for decision in range(60):
            # A nudge to one side for three decisions, then nothing, then a weak push back.
            table[shift, -1] = 1.0 if decision < 3 else (-0.3 if 30 <= decision < 40 else 0.0)
            made = adapter.decide(frozen, fixed, weights, state, frozen.propose(state), memory)
            memory = made['memory']
            outputs.append(float(made['output'][0, shift]))
        self.assertGreater(outputs[29], 0.9)      # held with no input
        self.assertGreater(outputs[59], 0.9)      # and through a weak push back
        table[shift, -1] = -3.0                   # a strong push back lets go
        for _ in range(20):
            made = adapter.decide(frozen, fixed, weights, state, frozen.propose(state), memory)
            memory = made['memory']
        self.assertLess(float(made['output'][0, shift]), -0.9)

    def test_steering_shifts_towards_the_cheaper_push(self):
        adapter = adapters.MemoryAdapter(2)
        table = adapter.table(adapter.steering(0.5))
        called = adapter.names()
        np.testing.assert_array_equal(table[:2], 0)                 # no gain
        self.assertEqual(table[2, called.index('cost change, number 0 up, decisions 1 to 5')], -0.5)
        self.assertEqual(table[3, called.index('cost change, number 1 down, decisions 11 to 15')], 0.5)
        self.assertEqual(np.count_nonzero(table), 12)

    def test_saved_and_loaded(self):
        frozen, state, predictor = made_up()
        adapter = adapters.MemoryAdapter(2, lengths=(3, 6), added_on=True)
        fixed = self.fixed(frozen, state, predictor, adapter)
        weights = np.random.default_rng(5).normal(size=adapter.count).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            file = pathlib.Path(folder) / 'adapter.npz'
            adapters.save(file, adapter, weights, fixed)
            loaded, loaded_weights, loaded_fixed = adapters.load(file, 2)
        self.assertEqual(loaded.describe(), adapter.describe())
        np.testing.assert_array_equal(loaded_weights, weights)
        np.testing.assert_array_equal(loaded_fixed['sizes'], fixed['sizes'])
        self.assertEqual(set(loaded_fixed['predictor']), set(predictor))


class Rolls(unittest.TestCase):

    def test_rolls(self):
        frozen, state, predictor = made_up(states=5)
        adapter = adapters.MemoryAdapter(2)
        fixed = Adapter().fixed(frozen, state, predictor, adapter)
        imagination = Imagination(frozen, adapter, fixed, state)
        zero = np.zeros(adapter.count, np.float32)
        unadapted = imagination.roll(zero, 8, seed=3)
        self.assertEqual(unadapted['cost'].shape, (8, 5))
        self.assertFalse(unadapted['moved'].any())
        np.testing.assert_array_equal(unadapted['size taken'], unadapted['size proposed'])
        # The same seed gives the same roll, and another seed another.
        np.testing.assert_array_equal(imagination.roll(zero, 8, seed=3)['reward'], unadapted['reward'])
        self.assertFalse(np.array_equal(imagination.roll(zero, 8, seed=4)['reward'], unadapted['reward']))
        weights = np.random.default_rng(7).normal(size=adapter.count).astype(np.float32)
        adapted = imagination.roll(weights, 8, seed=3)
        self.assertTrue(adapted['moved'].any())
        # A roll can start from a memory that is not empty, and then goes another way.
        full = np.full((5, adapter.remembered), 0.9, np.float32)
        self.assertFalse(np.array_equal(imagination.roll(weights, 8, seed=3, memory=full)['reward'], adapted['reward']))


class MadeUpPlayer:
    """The made-up agent with a made-up simulator: its own model with other noise, and a cost for one corner of the state."""

    task, steps_per_decision, action_numbers = 'made-up', 2, 2

    def __init__(self, checkpoint=None):
        self.frozen = MadeUpAgent()
        self.frozen.kept = lambda episode, entries=slice(None): {'z': episode['z'][entries]}
        self.frozen.written_out = lambda kept: kept
        self.frozen.rows = lambda kept: kept['z'].astype(np.float32)

    def play(self, seeds, adapter=None, memory=None, decisions=30):
        for seed in seeds:
            random = np.random.default_rng(seed)
            state = {'z': random.normal(size=(1, 6)).astype(np.float32)}
            episode = {name: [] for name in ('z', 'proposed', 'action', 'reward', 'cost', 'memory')}
            carried, reward, cost = memory, 0.0, 0.0
            for decision in range(decisions + 1):
                proposed = self.frozen.propose(state)
                action = proposed
                if adapter is not None and memory is None:
                    action = adapter(state, proposed)
                elif adapter is not None:
                    episode['memory'].append(carried[0])
                    action, carried = adapter(state, proposed, carried)
                for name, value in (('z', state['z'][0]), ('proposed', proposed[0]), ('action', action[0]),
                                    ('reward', reward), ('cost', cost)):
                    episode[name].append(value)
                following = np.tanh(state['z'] @ self.frozen._from_state + np.clip(action, -1, 1) @ self.frozen._from_action)
                state = {'z': (following + 0.05 * random.normal(size=following.shape)).astype(np.float32)}
                reward, cost = float(state['z'].sum()), float(state['z'][0, 1] > 0.6)
            if not episode['memory']:
                del episode['memory']
            yield {'seed': seed, **{name: np.array(values, np.float32) for name, values in episode.items()}}


class Search(unittest.TestCase):
    """search.py from end to end with the made-up agent: that every way of searching runs and leaves its files."""

    def run_search(self, *options):
        import contextlib
        import io
        import json
        import os
        import sys
        from unittest import mock

        import cost_predictor
        import search

        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            (folder / 'episodes').mkdir()
            for episode in MadeUpPlayer().play(range(1000, 1012)):
                np.savez(folder / 'episodes' / f'seed_{episode["seed"]:04d}.npz', **episode)
            (folder / 'episodes' / 'run.json').write_text(json.dumps({'checkpoint': 'made-up.ckpt', 'task': 'made-up'}))
            quiet = io.StringIO()
            with mock.patch.dict(os.environ, {'MS_RUN_DIR': str(folder / 'run')}), contextlib.redirect_stdout(quiet):
                (folder / 'run').mkdir()
                with mock.patch.object(sys, 'argv', ['cost_predictor.py', '--episodes', str(folder / 'episodes'),
                                                     '--test', '2', '--validation', '2', '--out', str(folder)]):
                    cost_predictor.main()
                with mock.patch.object(sys, 'argv', ['search.py', '--checkpoint', 'made-up.ckpt', '--predictor',
                                                     str(folder / 'cost_predictor.npz'), '--checks', '3', '--generations', '2',
                                                     '--episodes', '2', '--finalists', '2', '--more', '2', '--starts', '24',
                                                     '--roll', '6', '--candidates', '6', *options]), \
                        mock.patch.object(search, 'player_of', MadeUpPlayer):
                    search.main()
            final = json.loads((folder / 'run' / 'final.json').read_text())
            checks = [json.loads(line) for line in (folder / 'run' / 'checks.jsonl').read_text().splitlines()]
            adapter, weights, fixed = adapters.load(folder / 'run' / 'final.npz', 2)
            files = sorted(file.name for file in (folder / 'run').iterdir())
        self.assertEqual(len(checks), 3)
        self.assertEqual(final['generations'], 6)
        self.assertEqual(final['real episodes'], {'behind the first predictor': 10, 'played by the search': 10})
        self.assertEqual(weights.shape, (adapter.count,))
        return final, checks, adapter, files

    def test_one_score(self):
        final, checks, adapter, files = self.run_search()
        self.assertEqual(adapter.count, 152)
        self.assertNotIn('archive.jsonl', files)

    def test_one_score_cautious_and_damped_without_memory(self):
        final, checks, adapter, files = self.run_search('--cautious', '--damped', '--no-memory')
        self.assertEqual(adapter.count, 132)

    def test_archive(self):
        final, checks, adapter, files = self.run_search('--archive', '--diagonal')
        self.assertIn('archive.jsonl', files)

    def test_random_search_and_no_kept_lowest(self):
        self.run_search('--search', 'random', '--lowest', '0', '--added-on', '--no-stop')


class Cells(unittest.TestCase):

    def test_cells(self):
        from archive import COST_LEVELS, MANNERS, cell_of
        self.assertEqual(cell_of(1.0, 1.0), (0, MANNERS - 1))
        self.assertEqual(cell_of(1.7, 1.3), (0, MANNERS - 1))
        self.assertEqual(cell_of(0.6, 0.99), (0, MANNERS - 2))
        self.assertEqual(cell_of(0.5, 0.1), (1, 0))
        self.assertEqual(cell_of(0.26, 0.5), (1, 2))
        self.assertEqual(cell_of(0.03, 0.8), (5, 3))
        self.assertEqual(cell_of(0.0, 0.8), (COST_LEVELS - 1, 3))
        self.assertEqual(cell_of(1e-9, 0.8), (COST_LEVELS - 1, 3))

    def test_archive_and_what_the_search_is_told(self):
        from archive import Archive
        archive = Archive()
        archive.rebuild([{'weights': 0, 'reward': 1.0, 'cost': 1.0, 'kept': 1.0}])
        scores, added = archive.take_in([
            {'weights': 1, 'reward': 0.9, 'cost': 0.3, 'kept': 1.0},     # a new cell
            {'weights': 2, 'reward': 1.1, 'cost': 0.9, 'kept': 1.0},     # beats the one in its cell
            {'weights': 3, 'reward': 0.8, 'cost': 0.9, 'kept': 1.0},     # falls short
            {'weights': 4, 'reward': 0.95, 'cost': 0.3, 'kept': 1.0}])   # the same new cell, and better
        self.assertEqual(added, 3)
        self.assertEqual(len(archive), 2)
        self.assertGreater(scores[3], scores[0])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], 0)
        self.assertLess(scores[2], 0)
        self.assertEqual(archive.best(0.5)['weights'], 4)
        self.assertEqual(archive.best(0.1)['weights'], 4)      # none within: the least cost
        self.assertEqual(archive.best(1.0)['weights'], 2)
        self.assertEqual([member['weights'] for member in archive.near(0.2)], [4])
        # Scored again, two can fall into one cell, and one of them goes.
        archive.rebuild([{'weights': 2, 'reward': 1.0, 'cost': 0.4, 'kept': 1.0}, {'weights': 4, 'reward': 0.9, 'cost': 0.3, 'kept': 1.0}])
        self.assertEqual(len(archive), 1)

    def test_limit_level(self):
        from archive import LimitLevel
        level = LimitLevel(2, 50)
        self.assertAlmostEqual(level.level, 0.04)
        self.assertAlmostEqual(level.moved_by(2.5), 0.032)            # a quarter over the limit: down by that ratio
        self.assertAlmostEqual(LimitLevel(2, 50).moved_by(100), 0.04 / 1.5)
        self.assertAlmostEqual(LimitLevel(2, 50).moved_by(0), 0.06)
        self.assertAlmostEqual(LimitLevel(2, 50).moved_by(1.6), 0.05)
        damped = LimitLevel(2, 50, damped=True)
        damped.moved_by(100)
        self.assertAlmostEqual(damped.moved_by(0), 0.04 / 1.5 * 1.5 ** 0.5)


if __name__ == '__main__':
    unittest.main()
