"""
Tests for the lanes-fork group-aware island migration (lane_group_migration):
each lane group's best migrates, not just the globally top-scoring programs.
Off by default, so migrant selection is byte-identical to upstream.
"""

import random
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _prog(pid, lane=0.0, score=0.0, code=None):
    return Program(
        id=pid,
        code=code if code is not None else f"# {pid}\n",
        language="python",
        metrics={"combined_score": score, "lane": lane},
    )


def _db(group_migration=False, lane_metric="lane", num_islands=3, rate=0.1):
    config = Config()
    config.database.in_memory = True
    config.database.num_islands = num_islands
    config.database.lane_metric = lane_metric
    config.database.lane_group_migration = group_migration
    config.database.migration_rate = rate
    config.database.feature_dimensions = ["complexity", "diversity"]
    return ProgramDatabase(config.database)


class TestMigrantSelection(unittest.TestCase):
    def _island(self):
        # 8 high-scoring lane-0 + 2 low-scoring lane-1.
        return [_prog(f"q{i}", lane=0, score=0.5 + i) for i in range(8)] + [
            _prog("s0", lane=1, score=0.01),
            _prog("s1", lane=1, score=0.02),
        ]

    def test_off_selects_global_top_fraction(self):
        db = _db(group_migration=False, rate=0.1)
        migrants = db._select_migrants(self._island())
        # 10 programs * 0.1 -> 1 migrant, the global best (q7, score 7.5).
        self.assertEqual([m.id for m in migrants], ["q7"])

    def test_on_includes_every_group(self):
        db = _db(group_migration=True, rate=0.1)
        migrants = db._select_migrants(self._island())
        lanes = {m.metrics["lane"] for m in migrants}
        self.assertEqual(lanes, {0, 1})  # lane 1 propagates despite low scores
        # Group 0's migrant is its top scorer; group 1's is its best.
        self.assertIn("q7", [m.id for m in migrants])
        self.assertIn("s1", [m.id for m in migrants])  # 0.02 > 0.01

    def test_on_gives_a_singleton_group_a_migrant(self):
        db = _db(group_migration=True, rate=0.1)
        island = [_prog(f"q{i}", lane=0, score=i) for i in range(20)] + [_prog("lone", lane=1)]
        migrants = db._select_migrants(island)
        self.assertIn("lone", [m.id for m in migrants])  # rounds to 0 upstream, min 1 per group

    def test_off_matches_upstream_ordering(self):
        # With the flag off the result is exactly the global fitness-sorted top slice.
        db = _db(group_migration=False, rate=0.3)
        island = [_prog(f"p{i}", lane=i % 2, score=i) for i in range(10)]
        migrants = db._select_migrants(island)
        expected = sorted(island, key=lambda p: p.metrics["combined_score"], reverse=True)[:3]
        self.assertEqual([m.id for m in migrants], [p.id for p in expected])


class TestMigrationComposesWithSplitGrids(unittest.TestCase):
    def test_migrant_lands_under_group_prefix_no_silent_loss(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 2
        config.database.lane_metric = "lane"
        config.database.lane_group_migration = True
        config.database.lane_split_grids = True
        config.database.migration_rate = 0.5
        config.database.migration_interval = 1
        db = ProgramDatabase(config.database)
        # Distinct code -> distinct cells; lane 1 present only on island 0.
        db.add(_prog("a", lane=0, score=0.5, code="x = 1\n"), target_island=0)
        db.add(_prog("b", lane=1, score=0.4, code="y = 2\n" * 5), target_island=0)
        journal_before = len(db.eviction_journal)
        db.migrate_programs()
        # A lane-1 migrant reached island 1 under a group-1-prefixed key.
        keys_i1 = list(db.island_feature_maps[1].keys())
        self.assertTrue(any(k.startswith("1:") for k in keys_i1))
        # Any eviction during migration was journaled (no silent loss).
        for entry in db.eviction_journal[journal_before:]:
            self.assertIn("reason", entry)


class TestSamplingDeterminism(unittest.TestCase):
    """The RNG-audit regression: same seed + same adds -> same parent draws."""

    def _run(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 1
        config.database.lane_metric = "lane"
        config.database.lane_sampling_gamma = 3.0
        config.database.lane_rank_weight = 0.85
        config.database.random_seed = 20260823
        db = ProgramDatabase(config.database)
        for i in range(6):
            db.add(
                _prog(f"p{i}", lane=i % 2, score=0.1 * i, code=f"# p{i}\n{'x' * i}"),
                target_island=0,
            )
        random.seed(config.random_seed)  # reset global state the DB seeded at init
        return [db.sample()[0].id for _ in range(20)]

    def test_same_seed_same_parent_sequence(self):
        self.assertEqual(self._run(), self._run())


if __name__ == "__main__":
    unittest.main()
