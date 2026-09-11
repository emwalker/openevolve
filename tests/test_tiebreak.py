"""Tiebreak on exactly-equal fitness: which of two equals is kept.

With `tiebreak_metric` unset every path is upstream's, where a tie goes to the
incumbent; these tests are about what changes when it is set.
"""

import time
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase, tiebreak_of


def _program(pid, *, score, tiebreak=None):
    # `bucket` is constant, so every program shares one MAP-Elites cell and any
    # two of them must contest it.
    metrics = {"combined_score": score, "bucket": 0.5}
    if tiebreak is not None:
        metrics["size"] = tiebreak
    return Program(id=pid, code=f"# {pid}", metrics=metrics, timestamp=time.time())


def _database(**overrides):
    config = Config().database
    config.in_memory = True
    config.num_islands = 1
    for key, value in overrides.items():
        setattr(config, key, value)
    return ProgramDatabase(config)


class TestTiebreakPredicate(unittest.TestCase):
    def test_unconfigured_reads_nothing(self):
        self.assertIsNone(tiebreak_of(_program("a", score=0.1, tiebreak=3.0), None))

    def test_a_missing_or_unparseable_value_is_none(self):
        self.assertIsNone(tiebreak_of(_program("a", score=0.1), "size"))
        bad = Program(id="b", code="", metrics={"size": "wide"}, timestamp=time.time())
        self.assertIsNone(tiebreak_of(bad, "size"))

    def test_a_present_value_is_read_as_a_float(self):
        self.assertEqual(tiebreak_of(_program("a", score=0.1, tiebreak=3), "size"), 3.0)


class TestTiebreakComparison(unittest.TestCase):
    def test_unconfigured_a_tie_goes_to_the_incumbent(self):
        db = _database()
        a = _program("a", score=0.5, tiebreak=1.0)
        b = _program("b", score=0.5, tiebreak=9.0)
        self.assertFalse(db._is_better(a, b))
        self.assertFalse(db._is_better(b, a))

    def test_a_tie_is_broken_by_the_metric_lower_first(self):
        db = _database(tiebreak_metric="size")
        a = _program("a", score=0.5, tiebreak=1.0)
        b = _program("b", score=0.5, tiebreak=9.0)
        self.assertTrue(db._is_better(a, b))
        self.assertFalse(db._is_better(b, a))

    def test_the_direction_flag_reverses_it(self):
        db = _database(tiebreak_metric="size", tiebreak_lower_is_better=False)
        a = _program("a", score=0.5, tiebreak=1.0)
        b = _program("b", score=0.5, tiebreak=9.0)
        self.assertFalse(db._is_better(a, b))
        self.assertTrue(db._is_better(b, a))

    def test_an_unequal_fitness_ignores_the_metric(self):
        """The tiebreak orders equals; it never overturns a fitness difference,
        however small."""
        db = _database(tiebreak_metric="size")
        worse_but_simpler = _program("a", score=0.4, tiebreak=1.0)
        better_but_complex = _program("b", score=0.5, tiebreak=9.0)
        self.assertFalse(db._is_better(worse_but_simpler, better_but_complex))
        self.assertTrue(db._is_better(better_but_complex, worse_but_simpler))

    def test_a_program_with_no_measurement_loses_the_tie(self):
        db = _database(tiebreak_metric="size")
        measured = _program("a", score=0.5, tiebreak=9.0)
        unmeasured = _program("b", score=0.5)
        self.assertTrue(db._is_better(measured, unmeasured))
        self.assertFalse(db._is_better(unmeasured, measured))

    def test_two_unmeasured_or_two_equal_programs_fall_through(self):
        db = _database(tiebreak_metric="size")
        self.assertFalse(db._is_better(_program("a", score=0.5), _program("b", score=0.5)))
        a = _program("a", score=0.5, tiebreak=4.0)
        b = _program("b", score=0.5, tiebreak=4.0)
        self.assertFalse(db._is_better(a, b))
        self.assertFalse(db._is_better(b, a))


class TestTiebreakInTheFeatureMap(unittest.TestCase):
    """The comparator is shared, so the MAP-Elites cell inherits the preference:
    a simpler program of equal fitness takes the cell from a heavier one."""

    def _cell_holder(self, db):
        (key,) = db.island_feature_maps[0]
        return db.island_feature_maps[0][key]

    def test_the_simpler_of_two_equals_holds_the_cell(self):
        db = _database(tiebreak_metric="size", feature_dimensions=["bucket"], feature_bins=2)
        db.add(_program("complex", score=0.5, tiebreak=9.0))
        db.add(_program("simple", score=0.5, tiebreak=1.0))
        self.assertEqual(self._cell_holder(db), "simple")

    def test_the_heavier_of_two_equals_does_not_take_the_cell_back(self):
        db = _database(tiebreak_metric="size", feature_dimensions=["bucket"], feature_bins=2)
        db.add(_program("simple", score=0.5, tiebreak=1.0))
        db.add(_program("complex", score=0.5, tiebreak=9.0))
        self.assertEqual(self._cell_holder(db), "simple")


if __name__ == "__main__":
    unittest.main()
