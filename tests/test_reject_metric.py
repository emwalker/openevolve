"""A program the evaluator says it has already measured is not stored.

Storing it would put another program's numbers into a population slot, where it
is sampled as a parent and shown as an exemplar. `add` still returns the id, as
the novelty rejection does, so the caller's bookkeeping is unchanged.
"""

import time
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _program(pid, *, score=0.5, replayed=None):
    metrics = {"combined_score": score}
    if replayed is not None:
        metrics["replayed"] = replayed
    return Program(id=pid, code=f"# {pid}", metrics=metrics, timestamp=time.time())


def _database(**overrides):
    config = Config().database
    config.in_memory = True
    config.num_islands = 1
    for key, value in overrides.items():
        setattr(config, key, value)
    return ProgramDatabase(config)


class TestRejectMetric(unittest.TestCase):
    def setUp(self):
        self.db = _database(reject_metric="replayed")

    def test_a_marked_program_is_stored_nowhere(self):
        returned = self.db.add(_program("dupe", replayed=1.0))
        self.assertEqual(returned, "dupe", "the caller still gets an id back")
        self.assertNotIn("dupe", self.db.programs)
        self.assertNotIn("dupe", self.db.islands[0])
        self.assertNotIn("dupe", self.db.archive)
        self.assertNotIn("dupe", {pid for m in self.db.island_feature_maps for pid in m.values()})

    def test_an_unmarked_or_falsy_program_is_admitted(self):
        self.db.add(_program("fresh", replayed=0.0))
        self.db.add(_program("unmeasured"))
        self.assertIn("fresh", self.db.programs)
        self.assertIn("unmeasured", self.db.programs)

    def test_unconfigured_admits_everything(self):
        db = _database()
        db.add(_program("dupe", replayed=1.0))
        self.assertIn("dupe", db.programs)

    def test_a_rejected_program_does_not_become_the_best(self):
        self.db.add(_program("real", score=0.1))
        self.db.add(_program("dupe", score=0.9, replayed=1.0))
        best = self.db.get_best_program()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "real")

    def test_a_rejected_program_does_not_displace_a_cell_owner(self):
        """The failure this pins: a replay taking its twin's cell would evict
        the program it is a copy of."""
        self.db.add(_program("owner", score=0.5))
        before = {pid for m in self.db.island_feature_maps for pid in m.values()}
        self.db.add(_program("dupe", score=0.9, replayed=1.0))
        after = {pid for m in self.db.island_feature_maps for pid in m.values()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
