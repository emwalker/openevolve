"""
Tests for the lanes-fork additions to ProgramDatabase:
group-balanced eviction, the eviction journal, and declared feature domains.
"""

import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _prog(pid: str, lane: float, score: float) -> Program:
    return Program(
        id=pid,
        code=f"# {pid}",
        language="python",
        metrics={"combined_score": score, "lane": lane},
    )


class TestGroupBalancedEviction(unittest.TestCase):
    """_select_removals redirects eviction to the most-populous group.

    Tested at the selection-logic level (hand-built homeless/elite lists) so the
    MAP-Elites cell machinery and fitness averaging cannot confound the result.
    """

    def _db(self, lane_metric=None):
        config = Config()
        config.database.in_memory = True
        config.database.lane_metric = lane_metric
        return ProgramDatabase(config.database)

    def _scenario(self):
        # Group 0: one low-scoring member (the global worst).
        # Group 1: three higher-scoring members (the populous group).
        a_lo = _prog("a_lo", lane=0, score=0.01)
        b1 = _prog("b1", lane=1, score=0.90)
        b2 = _prog("b2", lane=1, score=0.91)
        b3 = _prog("b3", lane=1, score=0.92)
        non_elite = [a_lo, b1, b2, b3]  # worst-first, as _enforce_population_limit sorts
        return non_elite, [], non_elite  # (non_elite, elite, all_programs)

    def test_upstream_evicts_global_worst_when_no_lane_metric(self):
        db = self._db(lane_metric=None)
        non_elite, elite, allp = self._scenario()
        chosen = db._select_removals(1, non_elite, elite, allp)
        self.assertEqual([p.id for p in chosen], ["a_lo"])  # global worst

    def test_lane_metric_evicts_from_most_populous_group(self):
        db = self._db(lane_metric="lane")
        non_elite, elite, allp = self._scenario()
        chosen = db._select_removals(1, non_elite, elite, allp)
        # Group 1 is most populous, so its worst homeless goes, not group 0's
        # globally-lowest lone member.
        self.assertEqual([p.id for p in chosen], ["b1"])

    def test_lane_metric_falls_back_to_elite_when_homeless_exhausted(self):
        db = self._db(lane_metric="lane")
        non_elite, _, allp = self._scenario()
        elite = [_prog("e_owner", lane=1, score=0.5)]
        allp = non_elite + elite
        chosen = db._select_removals(5, non_elite, elite, allp)
        # All four homeless first (any order), then the elite owner last.
        self.assertEqual(len(chosen), 5)
        self.assertEqual(chosen[-1].id, "e_owner")


class TestEvictionJournal(unittest.TestCase):
    """Every removal is recorded, and the journal survives save/load."""

    def test_population_limit_removal_is_journaled(self):
        config = Config()
        config.database.in_memory = True
        config.database.population_size = 2
        db = ProgramDatabase(config.database)
        for pid, score in [("p0", 0.1), ("p1", 0.2), ("p2", 0.3)]:
            db.add(_prog(pid, 0, score), target_island=0)
        self.assertTrue(db.eviction_journal)
        entry = db.eviction_journal[0]
        self.assertEqual(entry["reason"], "population_limit")
        self.assertIn("combined_score", entry)
        self.assertIn("id", entry)
        self.assertIsInstance(entry["iteration"], int)  # stamped with when it happened

    def test_journal_iteration_reflects_the_add(self):
        config = Config()
        config.database.in_memory = True
        config.database.population_size = 2
        db = ProgramDatabase(config.database)
        db.add(_prog("p0", 0, 0.1), target_island=0)
        db.add(_prog("p1", 0, 0.2), target_island=0)
        db.add(_prog("p2", 0, 0.3), iteration=42, target_island=0)  # triggers the cull
        self.assertEqual(db.eviction_journal[-1]["iteration"], 42)

    def test_journal_survives_save_and_load(self):
        import tempfile

        config = Config()
        config.database.in_memory = True
        config.database.population_size = 2
        db = ProgramDatabase(config.database)
        for pid, score in [("p0", 0.1), ("p1", 0.2), ("p2", 0.3)]:
            db.add(_prog(pid, 0, score), target_island=0)
        n = len(db.eviction_journal)
        self.assertGreater(n, 0)
        with tempfile.TemporaryDirectory() as d:
            db.save(d, iteration=1)
            db2 = ProgramDatabase(config.database)
            db2.load(d)
        self.assertEqual(len(db2.eviction_journal), n)


class TestDeclaredFeatureDomains(unittest.TestCase):
    """A declared domain pins scaling and is not ratcheted by observed values."""

    def test_declared_domain_scales_against_pinned_range(self):
        config = Config()
        config.database.in_memory = True
        config.database.feature_domains = {"x": [0.0, 10.0]}
        db = ProgramDatabase(config.database)
        # Midpoint of the declared range scales to 0.5 regardless of stats.
        self.assertAlmostEqual(db._scale_feature_value("x", 5.0), 0.5)
        # Out-of-range clips.
        self.assertEqual(db._scale_feature_value("x", 20.0), 1.0)
        self.assertEqual(db._scale_feature_value("x", -5.0), 0.0)

    def test_declared_domain_does_not_ratchet(self):
        config = Config()
        config.database.in_memory = True
        config.database.feature_domains = {"x": [0.0, 10.0]}
        db = ProgramDatabase(config.database)
        db._update_feature_stats("x", 99.0)  # an outlier
        # Scaling still uses the declared range, not the outlier.
        self.assertNotIn("x", db.feature_stats)
        self.assertAlmostEqual(db._scale_feature_value("x", 5.0), 0.5)

    def test_undeclared_axis_uses_observed_range(self):
        config = Config()
        config.database.in_memory = True
        db = ProgramDatabase(config.database)
        db._update_feature_stats("y", 0.0)
        db._update_feature_stats("y", 10.0)
        self.assertAlmostEqual(db._scale_feature_value("y", 5.0), 0.5)


if __name__ == "__main__":
    unittest.main()
