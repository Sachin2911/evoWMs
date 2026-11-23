"""CMA-ES, the evolutionary search of the recipe, and a random search to set beside it.

Both search for a row of numbers and are used the same way. Each generation ask() gives
the candidates to try and tell() takes how good each one was, a higher score being better.
`centre` is the search's current best guess.

CMA-ES keeps a centre, a step size and a table of how the numbers vary together. It draws
its candidates around the centre, and moves all three towards the better candidates. It is
written after Hansen's tutorial (2016) and uses the settings given there.
"""

import numpy as np


class CMA:
    """CMA-ES from `centre`, with candidates first spread by `step` around it."""

    def __init__(self, centre, step, candidates=None, seed=0):
        n = len(centre)
        self.centre = np.asarray(centre, np.float64).copy()
        self.step = float(step)
        self.candidates = candidates or 4 + int(3 * np.log(n))
        # The better half of the candidates move the search, the best of them the most.
        self._parents = self.candidates // 2
        weights = np.log(self._parents + 0.5) - np.log(np.arange(1, self._parents + 1))
        self._weights = weights / weights.sum()
        effective = 1 / (self._weights ** 2).sum()
        self._effective = effective
        # How fast each running record follows what the newest generation says.
        self._path_rate = (4 + effective / n) / (n + 4 + 2 * effective / n)
        self._step_rate = (effective + 2) / (n + effective + 5)
        self._one_rate = 2 / ((n + 1.3) ** 2 + effective)
        self._many_rate = min(1 - self._one_rate, 2 * (effective - 2 + 1 / effective) / ((n + 2) ** 2 + effective))
        self._damping = 1 + 2 * max(0.0, np.sqrt((effective - 1) / (n + 1)) - 1) + self._step_rate
        # The length a row of n random numbers has on average.
        self._expected = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self._path, self._step_path = np.zeros(n), np.zeros(n)
        self._table = np.eye(n)
        self._generation = 0
        self._random = np.random.default_rng(seed)
        self._take_apart()

    def _take_apart(self):
        """Splits the table into its directions and how far it reaches along each."""
        self._table = (self._table + self._table.T) / 2
        reach, self._directions = np.linalg.eigh(self._table)
        self._reach = np.sqrt(np.maximum(reach, 1e-20))

    def ask(self):
        """The candidates of this generation, one row each."""
        draws = self._random.standard_normal((self.candidates, len(self.centre)))
        self._moves = (draws * self._reach) @ self._directions.T
        return self.centre + self.step * self._moves

    def tell(self, scores):
        """Takes the score of each candidate that ask() gave, in the same order."""
        n = len(self.centre)
        best = np.argsort(-np.asarray(scores, np.float64))[:self._parents]
        moves = self._moves[best]
        move = self._weights @ moves
        self.centre = self.centre + self.step * move
        self._generation += 1

        whitened = self._directions @ ((self._directions.T @ move) / self._reach)
        self._step_path = ((1 - self._step_rate) * self._step_path
                           + np.sqrt(self._step_rate * (2 - self._step_rate) * self._effective) * whitened)
        length = np.linalg.norm(self._step_path)
        # The table's own record is held still while the step size is growing fast.
        steady = length / np.sqrt(1 - (1 - self._step_rate) ** (2 * self._generation)) / self._expected < 1.4 + 2 / (n + 1)
        self._path = ((1 - self._path_rate) * self._path
                      + steady * np.sqrt(self._path_rate * (2 - self._path_rate) * self._effective) * move)
        from_path = np.outer(self._path, self._path) + (1 - steady) * self._path_rate * (2 - self._path_rate) * self._table
        from_candidates = (moves * self._weights[:, None]).T @ moves
        self._table = ((1 - self._one_rate - self._many_rate) * self._table
                       + self._one_rate * from_path + self._many_rate * from_candidates)
        self.step *= np.exp(self._step_rate / self._damping * (length / self._expected - 1))
        self._take_apart()


class RandomSearch:
    """Draws every candidate at random around the starting point and keeps the best one seen.

    The best so far is scored again in every generation, beside the new draws, because the
    scores of different generations are not measured on the same starting states.
    """

    def __init__(self, centre, step, candidates=16, seed=0):
        self._start = np.asarray(centre, np.float64).copy()
        self.centre = self._start.copy()
        self.step = float(step)
        self.candidates = candidates
        self._random = np.random.default_rng(seed)

    def ask(self):
        draws = self._random.standard_normal((self.candidates - 1, len(self.centre)))
        self._asked = np.concatenate([self.centre[None], self._start + self.step * draws])
        return self._asked

    def tell(self, scores):
        self.centre = self._asked[int(np.argmax(scores))].copy()
