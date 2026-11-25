"""Tests of the search's --settled option that need no agent, no simulator and no GPU.

Which states of an episode are kept as costly, and which a roll may start from, is plain
numpy. A made-up agent stands in whose state is one number, the entry's own index:

    python3 -m unittest test_settled
"""

import unittest

import numpy as np

from search import FURTHEST_BEFORE_A_COST, NEAREST_BEFORE_A_COST, before_a_cost, drawn, only, places, together


class MadeUpAgent:
    """Keeps one number per state, which is the entry of the episode it came from."""

    def kept(self, episode, entries):
        return {'z': episode['z'][entries]}

    def written_out(self, states):
        return states


def episode_with(charged, length=120, first=0):
    """An episode of `length` entries with a cost of 1 at the `charged` ones. Its states are numbered from `first`."""
    cost = np.zeros(length, np.float32)
    cost[list(charged)] = 1
    return {'cost': cost, 'z': first + np.arange(length)}


def drawn_as_before(frozen, latest, kept, count, random):
    """drawn() as it was before --settled, to compare with."""
    picked, left = [], count
    for states, share in kept:
        have = len(states['memory']) if states else 0
        number = min(int(share * count), have)
        if number:
            picked.append(only(states, random.choice(have, number, replace=False)))
            left -= number
    picked.append(only(latest, random.choice(len(latest['memory']), left, replace=len(latest['memory']) < left)))
    picked = together(picked)
    memory = picked.pop('memory')
    return frozen.written_out(picked), memory


class BeforeACost(unittest.TestCase):

    def test_the_numbers(self):
        self.assertEqual((NEAREST_BEFORE_A_COST, FURTHEST_BEFORE_A_COST), (10, 40))

    def test_no_cost_marks_nothing(self):
        self.assertFalse(before_a_cost(np.zeros(100)).any())

    def test_ten_to_forty_before_it_begins(self):
        marked = np.flatnonzero(before_a_cost(episode_with([70])['cost']))
        np.testing.assert_array_equal(marked, np.arange(30, 61))

    def test_a_cost_that_lasts_begins_once(self):
        marked = np.flatnonzero(before_a_cost(episode_with(range(70, 90))['cost']))
        np.testing.assert_array_equal(marked, np.arange(30, 61))

    def test_a_cost_in_the_first_entries(self):
        self.assertFalse(before_a_cost(episode_with([0])['cost']).any())
        self.assertFalse(before_a_cost(episode_with([9])['cost']).any())
        np.testing.assert_array_equal(np.flatnonzero(before_a_cost(episode_with([10])['cost'])), [0])
        np.testing.assert_array_equal(np.flatnonzero(before_a_cost(episode_with([25])['cost'])), np.arange(0, 16))

    def test_charged_entries_are_not_marked(self):
        # A cost at 30 to 32 lies in the stretch before the one that begins at 60.
        cost = episode_with([30, 31, 32, 60])['cost']
        marked = before_a_cost(cost)
        self.assertFalse(marked[cost > 0].any())
        expected = sorted((set(range(0, 21)) | set(range(20, 51))) - {30, 31, 32})
        np.testing.assert_array_equal(np.flatnonzero(marked), expected)


class Places(unittest.TestCase):

    empty = np.zeros((1, 2), np.float32)

    def test_without_settled_it_is_as_before(self):
        episode = episode_with([5, 50, 51])
        states, failed = places(MadeUpAgent(), episode, 3, self.empty)
        self.assertEqual(set(states), {'z', 'memory'})
        np.testing.assert_array_equal(np.flatnonzero(failed), [2, 3, 4, 5, 47, 48, 49, 50, 51])

    def test_settled(self):
        episode = episode_with([5, 50, 51])
        states, failed = places(MadeUpAgent(), episode, 3, self.empty, settled=True)
        self.assertEqual(len(states['z']), 119)
        np.testing.assert_array_equal(np.flatnonzero(states['charged']), [5, 50, 51])
        np.testing.assert_array_equal(np.flatnonzero(failed), np.arange(10, 41))

    def test_a_cost_at_the_last_entry_counts(self):
        states, failed = places(MadeUpAgent(), episode_with([119]), 3, self.empty, settled=True)
        self.assertFalse(states['charged'].any())
        np.testing.assert_array_equal(np.flatnonzero(failed), np.arange(79, 110))

    def test_an_episode_with_no_cost(self):
        states, failed = places(MadeUpAgent(), episode_with([]), 3, self.empty, settled=True)
        self.assertFalse(failed.any() or states['charged'].any())
        self.assertEqual(len(only(states, failed)['z']), 0)


class Drawn(unittest.TestCase):

    empty = np.zeros((1, 2), np.float32)

    def sets(self, settled):
        frozen = MadeUpAgent()
        # Three episodes, their states numbered from 0, 1000 and 2000.
        seen = [places(frozen, episode_with(charged, first=1000 * number), 3, self.empty, settled)
                for number, charged in enumerate(([5, 50, 51], range(60, 100), []))]
        latest = together([states for states, _ in seen])
        failures = together([only(states, failed) for states, failed in seen])
        return frozen, latest, failures, seen[1][0]

    def test_without_settled_it_draws_as_before(self):
        frozen, latest, failures, lowest = self.sets(False)
        for kept in ([(failures, 0.25), (lowest, 0.25)], [(failures, 0.25), (None, 0.25)]):
            for count in (64, 1000):
                now, memory_now = drawn(frozen, latest, kept, count, np.random.default_rng(3))
                then, memory_then = drawn_as_before(frozen, latest, kept, count, np.random.default_rng(3))
                self.assertEqual(set(now), set(then))
                np.testing.assert_array_equal(now['z'], then['z'])
                np.testing.assert_array_equal(memory_now, memory_then)

    def test_settled_starts_in_no_charged_state(self):
        frozen, latest, failures, lowest = self.sets(True)
        charged = {5, 50, 51} | set(range(1060, 1100))
        for seed in range(20):
            starts, memory = drawn(frozen, latest, [(failures, 0.25), (lowest, 0.25)], 64, np.random.default_rng(seed), True)
            self.assertEqual(set(starts), {'z'})
            self.assertEqual((len(starts['z']), len(memory)), (64, 64))
            # 16 from the kept costly states, 16 from the lowest episode, 32 from the latest.
            self.assertTrue(set(starts['z'][:16]) <= set(range(10, 41)) | set(range(1020, 1051)))
            self.assertTrue(set(starts['z'][16:32]) <= set(range(1000, 1060)) | set(range(1100, 1119)))
            self.assertFalse(set(starts['z']) & charged)

    def test_a_short_set_gives_what_it_has(self):
        frozen, latest, failures, lowest = self.sets(True)
        few = only(failures, np.arange(5))
        starts, _ = drawn(frozen, latest, [(few, 0.25), (None, 0.25)], 64, np.random.default_rng(0), True)
        self.assertEqual(len(starts['z']), 64)
        # More than the latest has uncharged: they are drawn with repeats.
        starts, _ = drawn(frozen, latest, [(few, 0.25)], 2000, np.random.default_rng(0), True)
        self.assertEqual(len(starts['z']), 2000)

    def test_latest_all_charged_is_drawn_from_as_it_is(self):
        frozen = MadeUpAgent()
        latest, _ = places(frozen, episode_with(range(120)), 3, self.empty, True)
        starts, _ = drawn(frozen, latest, [(None, 0.25)], 8, np.random.default_rng(0), True)
        self.assertEqual(len(starts['z']), 8)


class Search(unittest.TestCase):
    """search.py with --settled from end to end, with the made-up agent of test_memory.py."""

    def run_search(self, *options):
        import test_memory
        return test_memory.Search('run_search').run_search('--settled', '--roll', '14', *options)

    def test_one_score(self):
        final, checks, _, _ = self.run_search()
        self.assertTrue(final['settings']['settled'])
        self.assertTrue(all(np.isfinite(check['imagined cost share']) for check in checks))

    def test_archive(self):
        final, checks, _, _ = self.run_search('--archive')
        self.assertTrue(final['settings']['settled'])
        self.assertTrue(all(np.isfinite(check['imagined cost share']) for check in checks))

    def test_the_roll_has_to_be_longer_than_what_is_left_out(self):
        with self.assertRaises(SystemExit):
            self.run_search('--roll', '10')


if __name__ == '__main__':
    unittest.main()
