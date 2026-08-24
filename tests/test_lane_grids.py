"""
Tests for the lanes-fork per-group MAP-Elites grids (lane_split_grids):
group-prefixed cell keys give each group its own grid. Off by default, so an
unconfigured fork keys cells byte-for-byte as upstream.
"""

import tempfile
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _prog(pid, lane=0.0, score=0.0, tpy=0.0):
    return Program(
        id=pid,
        code=f"# {pid}",
        language="python",
        metrics={"combined_score": score, "lane": lane, "tpy": tpy},
    )


def _db(split=False, lane_metric="lane", num_islands=1, population_size=100):
    config = Config()
    config.database.in_memory = True
    config.database.num_islands = num_islands
    config.database.population_size = population_size
    config.database.lane_metric = lane_metric
    config.database.lane_split_grids = split
    # A single custom axis keeps cells easy to reason about; lane is the group.
    config.database.feature_dimensions = ["tpy"]
    config.database.feature_bins = 4
    return ProgramDatabase(config.database)


class TestKeyPrefixing(unittest.TestCase):
    def test_flag_off_is_byte_identical_to_upstream(self):
        db = _db(split=False)
        key = db._feature_coords_to_key([1, 2, 3], group=2)
        self.assertEqual(key, "1-2-3")  # group ignored when the flag is off

    def test_flag_on_prefixes_group(self):
        db = _db(split=True)
        self.assertEqual(db._feature_coords_to_key([1, 2, 3], group=2), "2:1-2-3")
        # group None still unprefixed (caller opted out)
        self.assertEqual(db._feature_coords_to_key([1, 2, 3], group=None), "1-2-3")


class TestSeparateGrids(unittest.TestCase):
    def test_same_axes_different_groups_occupy_different_cells(self):
        db = _db(split=True, num_islands=1)
        db.add(_prog("a", lane=0, score=0.5, tpy=1.0), target_island=0)
        db.add(_prog("b", lane=1, score=0.5, tpy=1.0), target_island=0)
        # Identical on the tpy axis but different groups -> both survive as owners.
        self.assertIn("a", db.programs)
        self.assertIn("b", db.programs)
        keys = list(db.island_feature_maps[0].keys())
        self.assertEqual(len(keys), 2)
        self.assertTrue(any(k.startswith("0:") for k in keys))
        self.assertTrue(any(k.startswith("1:") for k in keys))

    def test_same_group_same_axes_contest_one_cell(self):
        db = _db(split=True, num_islands=1)
        db.add(_prog("lo", lane=0, score=0.2, tpy=1.0), target_island=0)
        db.add(_prog("hi", lane=0, score=0.9, tpy=1.0), target_island=0)
        # Same group + same cell -> the better one owns it, the loser is displaced.
        self.assertIn("hi", db.programs)
        self.assertEqual(len(db.island_feature_maps[0]), 1)
        self.assertIn("hi", db.island_feature_maps[0].values())

    def test_shared_grid_collapses_the_same_two(self):
        # The contrast: without split grids, a and b (same axes, different group)
        # DO contest one cell, because lane is not part of the key.
        db = _db(split=False, num_islands=1)
        db.add(_prog("a", lane=0, score=0.5, tpy=1.0), target_island=0)
        db.add(_prog("b", lane=1, score=0.9, tpy=1.0), target_island=0)
        self.assertEqual(len(db.island_feature_maps[0]), 1)  # one cell, b won

    def test_missing_metric_group_gets_its_own_prefix(self):
        db = _db(split=True, num_islands=1)
        db.add(_prog("a", lane=0, score=0.5, tpy=1.0), target_island=0)
        p = Program(id="nom", code="# nom", language="python", metrics={"tpy": 1.0})
        db.add(p, target_island=0)  # no lane metric -> group -1
        keys = list(db.island_feature_maps[0].keys())
        self.assertTrue(any(k.startswith("-1:") for k in keys))
        self.assertIn("a", db.programs)
        self.assertIn("nom", db.programs)


class TestEvictionComposition(unittest.TestCase):
    def test_split_grids_compose_with_the_cull(self):
        # Split grids must not break the population cull: an overflow still culls,
        # respects the cap, journals the removal, and never drops the just-added
        # program. (Group-balanced targeting itself is tranche 1's concern and
        # needs homeless programs, which distinct cell owners are not.)
        db = _db(split=True, num_islands=1, population_size=3)
        for i, tpy in enumerate([1.0, 2.0]):
            db.add(_prog(f"q{i}", lane=0, score=0.5 + i, tpy=tpy), target_island=0)
        db.add(_prog("s0", lane=1, score=0.4, tpy=1.0), target_island=0)
        self.assertEqual(len(db.programs), 3)
        journal_before = len(db.eviction_journal)
        db.add(_prog("q2", lane=0, score=3.0, tpy=3.0), target_island=0)
        self.assertLessEqual(len(db.programs), 3)  # cap respected
        self.assertIn("q2", db.programs)  # just-added protected
        self.assertTrue(db.eviction_journal[journal_before:])  # removal journaled


class TestRoundTrip(unittest.TestCase):
    def test_prefixed_maps_survive_save_load(self):
        db = _db(split=True, num_islands=1)
        db.add(_prog("a", lane=0, score=0.5, tpy=1.0), target_island=0)
        db.add(_prog("b", lane=1, score=0.5, tpy=1.0), target_island=0)
        with tempfile.TemporaryDirectory() as d:
            db.save(d, iteration=1)
            db2 = _db(split=True, num_islands=1)
            db2.load(d)
        self.assertEqual(
            set(db2.island_feature_maps[0].keys()), set(db.island_feature_maps[0].keys())
        )

    def test_load_warns_on_unmigrated_checkpoint_under_split_grids(self):
        # Save under the shared grid (bare keys), reload under split grids: the
        # load must warn that the checkpoint predates per-group grids.
        shared = _db(split=False, num_islands=1)
        shared.add(_prog("a", lane=0, score=0.5, tpy=1.0), target_island=0)
        with tempfile.TemporaryDirectory() as d:
            shared.save(d, iteration=1)
            split = _db(split=True, num_islands=1)
            with self.assertLogs("openevolve.database", level="WARNING") as cm:
                split.load(d)
        self.assertTrue(any("un-prefixed cell keys" in m for m in cm.output))


if __name__ == "__main__":
    unittest.main()
