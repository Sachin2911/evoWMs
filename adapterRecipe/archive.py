"""The archive: the best adapter found for each cost level and each manner of acting.

A search with an archive keeps many adapters and not one
(implementation/adapterArchitecture.md, decision 10). Every adapter that is scored falls
into a cell, and a cell keeps the adapter with the most imagined reward:

    cost level    the adapter's imagined cost as a share of the unadapted agent's. Each
                  level is half the one above: level 0 is everything above a half, level 1
                  from a quarter to a half, and so on. The last holds everything below.
    manner        how much of the proposed action's size the adapter keeps: four quarters
                  below 1, and a fifth cell for 1 or more. An adapter that brakes keeps
                  little, and one that redirects keeps about all of it.

The cost limit is a real cost, and the cost level is an imagined one. `LimitLevel` is the
imagined cost level that real episodes say matches the limit, and the current best is the
adapter with the most imagined reward at or under it.

Why it is built this way:

  The cost levels are in equal ratios because the limit is a few percent of the unadapted
  agent's cost. Equal steps between 0 and 1 would put every safe adapter in one cell.

  Every adapter the archive holds is scored again on each generation's starting states,
  and the archive is put together afresh from those scores. An adapter that was lucky on
  the states of its own generation would otherwise stay in its cell.

  What the search is told about a candidate is how it changed the archive: one that
  filled an empty cell counts most, then one that beat the adapter in its cell, by how
  much, then the others by how far short they fell. A search that is only told to lower
  cost goes for braking, which is the nearest way down; one that is told this is
  rewarded for every cell, the redirecting ones among them.

  Being told that alone, a search has no reason to go down in cost: it is rewarded as
  much for a little more reward in a cell far above the limit. So a search starts again,
  after every round of real episodes, from an adapter near the limit level (near()).
  The starts are what take the archive down to the limit, and what it is told spreads it
  over the manners of acting there.
"""

import numpy as np

COST_LEVELS = 8
MANNERS = 5


def cell_of(cost_share, kept_share):
    """The cell of an adapter with this imagined cost share and this share of the action's size kept."""
    if not cost_share > 0:
        level = COST_LEVELS - 1
    else:
        level = int(np.clip(np.floor(-np.log2(cost_share)), 0, COST_LEVELS - 1))
    manner = int(np.clip(np.floor(kept_share * (MANNERS - 1)), 0, MANNERS - 1))
    return level, manner


class Archive:
    """The adapters kept, one per cell. Each is a dict with `weights`, `reward`, `cost` and `kept`."""

    def __init__(self):
        self.members = {}

    def __len__(self):
        return len(self.members)

    def rebuild(self, rescored):
        """Puts the archive together afresh from its own adapters, scored again on new starting states.

        Two that now fall into one cell compete for it, so the archive can shrink.
        """
        self.members = {}
        for member in rescored:
            self._offer(member)

    def _offer(self, member):
        cell = cell_of(member['cost'], member['kept'])
        if cell not in self.members or member['reward'] > self.members[cell]['reward']:
            self.members[cell] = member
            return True
        return False

    def take_in(self, candidates):
        """Offers a generation's candidates to the archive; returns what the search is told about each.

        The score of a candidate is higher the more it changed the archive: 2,000 and its
        reward if it filled a cell that was empty before this generation, 1,000 and its
        lead if it beat the adapter that held its cell, and otherwise how far it fell
        short, which is below zero. The second number of what comes back is how many
        candidates the archive kept.
        """
        before = {cell: member['reward'] for cell, member in self.members.items()}
        scores, added = [], 0
        for candidate in candidates:
            cell = cell_of(candidate['cost'], candidate['kept'])
            if cell not in before:
                scores.append(2000 + candidate['reward'])
            elif candidate['reward'] > before[cell]:
                scores.append(1000 + candidate['reward'] - before[cell])
            else:
                scores.append(candidate['reward'] - before[cell])
            added += self._offer(candidate)
        return scores, added

    def best(self, limit_level):
        """The current best: the most imagined reward among the adapters at or under the limit level.

        If none is, the one with the least imagined cost.
        """
        within = [member for member in self.members.values() if member['cost'] <= limit_level]
        if within:
            return max(within, key=lambda member: member['reward'])
        return min(self.members.values(), key=lambda member: member['cost'])

    def near(self, limit_level):
        """The adapters a search may start again from: those at most one cost level above the limit level.

        Those just above the limit are the ones a search can bring under it. If there is
        none that low, those of the lowest cost level that holds any, which are the
        nearest the archive has come to the limit.
        """
        close = [member for member in self.members.values() if member['cost'] <= 2 * limit_level]
        if close:
            return close
        lowest = max(level for level, _ in self.members)
        return [member for (level, _), member in self.members.items() if level == lowest]

    def table(self):
        """The archive as rows for a record: the cell, and the reward, cost and kept share of its adapter."""
        return [{'cost level': cell[0], 'manner': cell[1], 'reward': member['reward'], 'cost': member['cost'],
                 'kept': member['kept']} for cell, member in sorted(self.members.items())]


class LimitLevel:
    """The imagined cost level that real episodes say matches the cost limit.

    It starts at the limit as a share of the unadapted agent's real cost. After each round
    of real episodes it moves by the ratio of the limit to the real cost of the adapter
    that played, by no more than a factor of `most`: down if those episodes cost more than
    the limit, up if less. With `damped`, that factor shrinks to its square root each time
    the real cost crosses the limit, so the level settles and does not swing.
    """

    def __init__(self, limit, unadapted_cost, most=1.5, damped=False):
        self.limit = limit
        self.level = min(1.0, limit / unadapted_cost) if unadapted_cost > 0 else 1.0
        self.most, self.damped = most, damped
        self._was_beyond = None

    def moved_by(self, real_cost):
        """Takes the real cost of the adapter that played; returns the new level."""
        beyond = real_cost > self.limit
        if self.damped and self._was_beyond is not None and beyond != self._was_beyond:
            self.most = float(np.sqrt(self.most))
        self._was_beyond = beyond
        ratio = self.most if real_cost <= 0 else self.limit / real_cost
        self.level = float(min(1.0, self.level * np.clip(ratio, 1 / self.most, self.most)))
        return self.level
