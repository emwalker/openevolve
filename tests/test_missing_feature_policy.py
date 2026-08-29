"""
Tests for the lanes-fork missing-feature policy (missing_feature_policy):
a program lacking a declared feature dimension can bin at the axis floor
instead of raising. Defaults to "error", so an unconfigured fork raises
exactly as upstream.
"""

import tempfile
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _prog(pid, metrics):
    return Program(id=pid, code=f"# {pid}", language="python", metrics=metrics)


def _db(policy=None, domains=None):
    config = Config()
    config.database.in_memory = True
    config.database.num_islands = 1
    config.database.feature_dimensions = ["turnover", "gross"]
    config.database.feature_bins = 4
    if domains is not None:
        config.database.feature_domains = domains
    if policy is not None:
        config.database.missing_feature_policy = policy
    return ProgramDatabase(config.database)


class TestDefaultIsUpstream(unittest.TestCase):
    def test_default_policy_is_error(self):
        self.assertEqual(Config().database.missing_feature_policy, "error")

    def test_missing_dimension_raises_by_default(self):
        db = _db()
        with self.assertRaises(ValueError) as ctx:
            db._calculate_feature_coords(_prog("a", {"turnover": 1.0}))
        self.assertIn("gross", str(ctx.exception))

    def test_complete_program_is_unaffected(self):
        db = _db()
        coords = db._calculate_feature_coords(_prog("a", {"turnover": 1.0, "gross": 0.5}))
        self.assertEqual(len(coords), 2)


class TestFloorPolicy(unittest.TestCase):
    def test_missing_dimension_bins_at_zero_without_a_domain(self):
        db = _db(policy="floor")
        coords = db._calculate_feature_coords(_prog("a", {"turnover": 1.0}))
        self.assertEqual(len(coords), 2)
        self.assertEqual(coords[1], 0)

    def test_missing_dimension_bins_at_the_declared_domain_floor(self):
        # An axis whose domain starts above zero must bin at its own floor, not
        # at a bin the declared range does not contain.
        db = _db(policy="floor", domains={"gross": [8.0, 88.0]})
        coords = db._calculate_feature_coords(_prog("a", {"turnover": 1.0}))
        self.assertEqual(coords[1], 0)

    def test_present_coordinates_are_untouched(self):
        db = _db(policy="floor")
        full = db._calculate_feature_coords(_prog("a", {"turnover": 3.0, "gross": 0.9}))
        partial = db._calculate_feature_coords(_prog("b", {"turnover": 3.0}))
        self.assertEqual(full[0], partial[0])

    def test_a_program_with_no_metrics_still_bins(self):
        db = _db(policy="floor")
        coords = db._calculate_feature_coords(_prog("a", {}))
        self.assertEqual(coords, [0, 0])

    def test_floor_policy_logs_a_warning(self):
        db = _db(policy="floor")
        with self.assertLogs("openevolve.database", level="WARNING") as logs:
            db._calculate_feature_coords(_prog("a", {"turnover": 1.0}))
        self.assertTrue(any("gross" in line for line in logs.output))

    def test_a_timed_out_program_can_be_added(self):
        # The case this exists for: a harness stage timeout yields a metrics dict
        # carrying none of the declared axes, and the whole iteration was lost.
        db = _db(policy="floor")
        db.add(_prog("timed-out", {"stage1_passed": 0.0, "error": 0.0}))
        self.assertIn("timed-out", db.programs)


if __name__ == "__main__":
    unittest.main()
