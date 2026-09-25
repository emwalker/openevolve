"""Constraint handling: infeasible programs are archived but never bred from.

With `feasibility_metric` unset every path is upstream's, which the rest of the
suite already covers; these tests are about what changes when it is set.
"""

import random
import time
import unittest

from openevolve.config import Config
from openevolve.database import (
    Program,
    ProgramDatabase,
    _group_score,
    is_feasible,
    rank_exemplars,
    violation_of,
)
from openevolve.utils.metrics_utils import get_fitness_score


def _program(pid, *, score, feasible=None, violation=None):
    metrics = {"combined_score": score}
    if feasible is not None:
        metrics["feas"] = feasible
    if violation is not None:
        metrics["viol"] = violation
    return Program(id=pid, code=f"# {pid}", metrics=metrics, timestamp=time.time())


def _database(**overrides):
    config = Config().database
    config.in_memory = True
    config.num_islands = 1
    for key, value in overrides.items():
        setattr(config, key, value)
    return ProgramDatabase(config)


class TestFeasibilityPredicates(unittest.TestCase):
    def test_unconfigured_everything_is_feasible(self):
        self.assertTrue(is_feasible(_program("a", score=0.1), None))
        self.assertEqual(violation_of(_program("a", score=0.1), None), 0.0)

    def test_a_missing_measurement_is_not_a_violation(self):
        """A program not measured on the constraint is not thereby in breach;
        it is the absent *violation* that ranks worst."""
        self.assertTrue(is_feasible(_program("a", score=0.1), "feas"))
        self.assertEqual(violation_of(_program("a", score=0.1), "viol"), float("inf"))

    def test_zero_is_infeasible_and_positive_is_feasible(self):
        self.assertFalse(is_feasible(_program("a", score=0.1, feasible=0.0), "feas"))
        self.assertTrue(is_feasible(_program("b", score=0.1, feasible=1.0), "feas"))


class TestFeasibleSelection(unittest.TestCase):
    def setUp(self):
        self.db = _database(feasibility_metric="feas", violation_metric="viol")

    def test_an_infeasible_program_is_stored_but_not_sampled(self):
        """The archive keeps the record; the gene pool does not."""
        self.db.add(_program("good", score=0.1, feasible=1.0, violation=0.0))
        self.db.add(_program("bad", score=0.9, feasible=0.0, violation=2.0))

        self.assertIn("bad", self.db.programs, "lineage must survive")
        for _ in range(40):
            self.assertEqual(self.db._sample_parent().id, "good")

    def test_the_best_program_is_never_an_infeasible_one(self):
        self.db.add(_program("good", score=0.1, feasible=1.0, violation=0.0))
        self.db.add(_program("bad", score=0.9, feasible=0.0, violation=2.0))
        self.assertEqual(self.db.get_best_program().id, "good")

    def test_with_nothing_feasible_the_closest_is_preferred(self):
        """A population that has not yet produced a feasible program still has a
        gradient: selection narrows to the half nearest feasibility."""
        for i, viol in enumerate([9.0, 8.0, 0.5, 0.6]):
            self.db.add(_program(f"p{i}", score=0.5, feasible=0.0, violation=viol))
        sampled = {self.db._sample_parent().id for _ in range(60)}
        self.assertTrue(sampled <= {"p2", "p3"}, f"sampled the far half too: {sampled}")

    def test_feasible_outranks_a_higher_scoring_infeasible_one(self):
        better = _program("good", score=0.1, feasible=1.0, violation=0.0)
        worse = _program("bad", score=0.9, feasible=0.0, violation=2.0)
        self.assertTrue(self.db._is_better(better, worse))
        self.assertFalse(self.db._is_better(worse, better))

    def test_two_infeasible_compare_by_violation(self):
        near = _program("near", score=0.1, feasible=0.0, violation=0.5)
        far = _program("far", score=0.9, feasible=0.0, violation=5.0)
        self.assertTrue(self.db._is_better(near, far))
        self.assertFalse(self.db._is_better(far, near))


class TestMembershipSurvivesSampling(unittest.TestCase):
    """Sampling declines to breed from an infeasible program; it does not evict
    it. Only a program absent from `programs` is stale."""

    def setUp(self):
        self.db = _database(feasibility_metric="feas", violation_metric="viol")

    def test_an_infeasible_program_keeps_its_island_across_sampling(self):
        for i in range(6):
            self.db.add(_program(f"p{i}", score=0.1 * i, feasible=0.0, violation=float(i + 1)))
        before = set(self.db.islands[0])
        for _ in range(20):
            self.db._sample_parent()
        self.assertEqual(set(self.db.islands[0]), before)

    def test_a_feasible_program_does_not_empty_its_island(self):
        """The failure this pins: one feasible arrival used to purge every
        infeasible member, collapsing selection onto a single lineage."""
        for i in range(5):
            self.db.add(_program(f"p{i}", score=0.1, feasible=0.0, violation=float(i + 1)))
        self.db.add(_program("ok", score=0.1, feasible=1.0, violation=0.0))
        for _ in range(20):
            self.db._sample_parent()
        self.assertEqual(len(self.db.islands[0]), 6)

    def test_an_infeasible_program_keeps_its_archive_membership(self):
        self.db.add(_program("ok", score=0.9, feasible=1.0, violation=0.0))
        self.db.add(_program("bad", score=0.5, feasible=0.0, violation=1.0))
        self.db.archive.update({"ok", "bad"})
        for _ in range(20):
            self.db._sample_exploitation_parent()
        self.assertIn("bad", self.db.archive)

    def test_a_missing_program_is_still_purged(self):
        """Upstream's actual cleanup survives: an id with no program is stale."""
        self.db.add(_program("ok", score=0.1, feasible=1.0, violation=0.0))
        self.db.islands[0].add("ghost")
        self.db.archive.update({"ok", "ghost"})
        for _ in range(30):  # _sample_parent picks a mode at random
            self.db._sample_parent()
        self.db._sample_exploitation_parent()
        self.assertNotIn("ghost", self.db.islands[0])
        self.assertNotIn("ghost", self.db.archive)


class TestMinimumBreedablePool(unittest.TestCase):
    def _populate(self, db):
        db.add(_program("ok", score=0.1, feasible=1.0, violation=0.0))
        for i, viol in enumerate([0.5, 0.6, 4.0, 9.0]):
            db.add(_program(f"p{i}", score=0.5, feasible=0.0, violation=viol))
        return db

    def test_min_pool_zero_is_the_hard_wall(self):
        db = self._populate(_database(feasibility_metric="feas", violation_metric="viol"))
        self.assertEqual(db._breedable_ids(list(db.programs)), ["ok"])

    def test_min_pool_tops_up_with_the_closest_infeasible(self):
        db = self._populate(
            _database(feasibility_metric="feas", violation_metric="viol", feasibility_min_pool=3)
        )
        self.assertEqual(db._breedable_ids(list(db.programs)), ["ok", "p0", "p1"])

    def test_min_pool_larger_than_the_scope_takes_everything(self):
        db = self._populate(
            _database(feasibility_metric="feas", violation_metric="viol", feasibility_min_pool=99)
        )
        self.assertEqual(set(db._breedable_ids(list(db.programs))), set(db.programs))

    def test_min_pool_never_reports_an_infeasible_best(self):
        db = _database(feasibility_metric="feas", violation_metric="viol", feasibility_min_pool=4)
        db.add(_program("ok", score=0.1, feasible=1.0, violation=0.0))
        db.add(_program("bad", score=0.9, feasible=0.0, violation=0.1))
        self.assertEqual(db.get_best_program().id, "ok")

    def test_without_a_lane_metric_the_pool_is_the_island_pool(self):
        """The ungrouped path is untouched: one pool, one top-up."""
        db = self._populate(
            _database(feasibility_metric="feas", violation_metric="viol", feasibility_min_pool=3)
        )
        self.assertEqual(db._breedable_ids(list(db.programs)), ["ok", "p0", "p1"])

    def test_min_pool_does_not_change_the_all_infeasible_fallback(self):
        """With nothing feasible the closest-half rule owns the scope; the floor
        is a top-up for the feasible branch only."""
        walled = _database(feasibility_metric="feas", violation_metric="viol")
        floored = _database(
            feasibility_metric="feas", violation_metric="viol", feasibility_min_pool=4
        )
        for db in (walled, floored):
            for i, viol in enumerate([9.0, 8.0, 0.5, 0.6]):
                db.add(_program(f"p{i}", score=0.1 * i, feasible=0.0, violation=viol))
        ids = ["p0", "p1", "p2", "p3"]
        got = floored._breedable_ids(ids)
        self.assertEqual(got, walled._breedable_ids(ids))
        self.assertTrue(set(got) <= {"p2", "p3"}, f"reached past the closest half: {got}")

    def test_min_pool_is_inert_without_a_violation_metric(self):
        db = _database(feasibility_metric="feas", feasibility_min_pool=4)
        db.add(_program("ok", score=0.1, feasible=1.0))
        db.add(_program("bad", score=0.9, feasible=0.0))
        self.assertEqual(db._breedable_ids(list(db.programs)), ["ok"])


class TestUnconfiguredIsUpstream(unittest.TestCase):
    def test_selection_ignores_the_metrics_when_unconfigured(self):
        db = _database()
        db.add(_program("good", score=0.1, feasible=1.0, violation=0.0))
        db.add(_program("bad", score=0.9, feasible=0.0, violation=2.0))
        # Highest score wins, exactly as upstream, feasibility notwithstanding.
        self.assertEqual(db.get_best_program().id, "bad")

    def test_breedable_is_the_identity_when_unconfigured(self):
        db = _database()
        ids = ["a", "b", "c"]
        for pid in ids:
            db.add(_program(pid, score=0.1, feasible=0.0, violation=5.0))
        self.assertEqual(db._breedable_ids(ids), ids)

    def test_the_parent_sequence_is_seed_stable_when_unconfigured(self):
        """Upstream sampling is untouched: a fixed seed reproduces its own run."""

        def sequence():
            random.seed(7)
            db = _database()
            for i in range(5):
                db.add(_program(f"p{i}", score=0.1 * i, feasible=0.0, violation=float(i)))
            ids = [db._sample_parent().id for _ in range(30)]
            return ids, set(db.islands[0])

        self.assertEqual(sequence(), sequence())


class TestExemplarOrder(unittest.TestCase):
    """What the model is SHOWN, not what the engine breeds from. Selection has
    ranked feasibility first since constraint handling landed; the display
    lists were still ordered on fitness alone."""

    def setUp(self):
        self.db = _database(feasibility_metric="feas", violation_metric="viol")

    def test_a_feasible_program_precedes_a_higher_scoring_infeasible_one(self):
        good = _program("good", score=0.1, feasible=1.0, violation=0.0)
        bad = _program("bad", score=0.9, feasible=0.0, violation=1.0)
        self.assertEqual([p.id for p in self.db.exemplar_order([bad, good])], ["good", "bad"])

    def test_feasible_programs_keep_their_fitness_order(self):
        lo = _program("lo", score=0.1, feasible=1.0, violation=0.0)
        hi = _program("hi", score=0.8, feasible=1.0, violation=0.0)
        self.assertEqual([p.id for p in self.db.exemplar_order([lo, hi])], ["hi", "lo"])

    def test_infeasible_programs_are_ordered_by_violation(self):
        near = _program("near", score=0.1, feasible=0.0, violation=0.5)
        far = _program("far", score=0.9, feasible=0.0, violation=5.0)
        self.assertEqual([p.id for p in self.db.exemplar_order([far, near])], ["near", "far"])

    def test_equal_violation_falls_back_to_fitness(self):
        lo = _program("lo", score=0.1, feasible=0.0, violation=2.0)
        hi = _program("hi", score=0.8, feasible=0.0, violation=2.0)
        self.assertEqual([p.id for p in self.db.exemplar_order([lo, hi])], ["hi", "lo"])

    def test_unconfigured_is_exactly_the_fitness_sort(self):
        """An unconfigured fork must order these lists as upstream does."""
        db = _database()
        programs = [
            _program("a", score=0.3, feasible=0.0, violation=9.0),
            _program("b", score=0.9, feasible=0.0, violation=1.0),
            _program("c", score=0.1, feasible=1.0, violation=0.0),
        ]
        expected = sorted(
            programs,
            key=lambda p: get_fitness_score(p.metrics, db.config.feature_dimensions),
            reverse=True,
        )
        self.assertEqual([p.id for p in db.exemplar_order(programs)], [p.id for p in expected])

    def test_rank_exemplars_honours_a_caller_supplied_score(self):
        """Each call site keeps its own notion of fitness; only the feasibility
        split is shared."""
        lo = _program("lo", score=0.1, feasible=1.0, violation=0.0)
        hi = _program("hi", score=0.8, feasible=1.0, violation=0.0)
        order = rank_exemplars([lo, hi], "feas", "viol", None, score=lambda p: -_group_score(p))
        self.assertEqual([p.id for p in order], ["lo", "hi"])

    def test_get_top_programs_never_leads_with_an_infeasible_one(self):
        self.db.add(_program("good", score=0.1, feasible=1.0, violation=0.0))
        self.db.add(_program("bad", score=0.9, feasible=0.0, violation=1.0))
        self.assertEqual([p.id for p in self.db.get_top_programs(n=2)], ["good", "bad"])

    def test_an_explicit_metric_still_means_that_metric(self):
        """`get_top_programs(metric=...)` answers a question about that metric;
        only the default ranking is the exemplar order."""
        self.db.add(_program("good", score=0.1, feasible=1.0, violation=0.0))
        self.db.add(_program("bad", score=0.9, feasible=0.0, violation=1.0))
        ranked = self.db.get_top_programs(n=2, metric="combined_score")
        self.assertEqual([p.id for p in ranked], ["bad", "good"])

    def test_island_best_score_is_the_ranked_best_not_the_highest(self):
        """The log line prints this number beside the feasibility-aware best
        program's id; they have to be the same program."""
        self.db.add(_program("good", score=0.1, feasible=1.0, violation=0.0))
        self.db.add(_program("bad", score=0.9, feasible=0.0, violation=1.0))
        stats = self.db.get_island_stats()[0]
        self.assertAlmostEqual(stats["best_score"], 0.1)

    def test_unconfigured_island_best_score_is_the_maximum(self):
        db = _database()
        db.add(_program("good", score=0.1, feasible=1.0, violation=0.0))
        db.add(_program("bad", score=0.9, feasible=0.0, violation=1.0))
        self.assertAlmostEqual(db.get_island_stats()[0]["best_score"], 0.9)


if __name__ == "__main__":
    unittest.main()


class TestGroupedBreedablePool(unittest.TestCase):
    """With `lane_metric` set the floor applies per group.

    A group with no feasible member is reachable through its own closest-half
    fallback; a group's top-up is never spent on another group's near-misses.
    Without the grouping, one group holding every feasible member silences the
    rest whatever the group-balanced sampler is set to.
    """

    def _grouped(self, **overrides):
        return _database(
            feasibility_metric="feas",
            violation_metric="viol",
            lane_metric="lane",
            **overrides,
        )

    @staticmethod
    def _member(pid, lane, *, feasible, violation, score=0.1):
        program = _program(pid, score=score, feasible=feasible, violation=violation)
        program.metrics["lane"] = lane
        return program

    @classmethod
    def _hold(cls, db, pid, lane, **kwargs):
        """Put a member in `db.programs` directly.

        `_breedable_ids` reads that mapping alone; going through `add()` would
        route the fixture through MAP-Elites eviction, which is a different
        rule and drops members whose cells collide.
        """
        db.programs[pid] = cls._member(pid, lane, **kwargs)

    def test_with_a_lane_metric_each_group_keeps_its_own_pool(self):
        db = self._grouped(feasibility_min_pool=4)
        for i in range(4):
            self._hold(db, f"a{i}", 0, feasible=1.0, violation=0.0)
        self._hold(db, "b0", 1, feasible=0.0, violation=1.0)
        self._hold(db, "b1", 1, feasible=0.0, violation=2.0)

        breedable = db._breedable_ids(list(db.programs))

        # Group A's four feasible meet the floor exactly, so its top-up is
        # empty; group B has none feasible and falls back to its closest half.
        self.assertEqual(set(breedable), {"a0", "a1", "a2", "a3", "b0"})
        self.assertTrue(any(pid.startswith("b") for pid in breedable))

    def test_a_group_with_a_feasible_member_tops_up_from_its_own_group(self):
        db = self._grouped(feasibility_min_pool=2)
        self._hold(db, "a_ok", 0, feasible=1.0, violation=0.0)
        for i, viol in enumerate([0.5, 3.0, 9.0]):
            self._hold(db, f"a{i}", 0, feasible=0.0, violation=viol)
        self._hold(db, "b_ok", 1, feasible=1.0, violation=0.0)
        for i, viol in enumerate([0.2, 4.0, 8.0]):
            self._hold(db, f"b{i}", 1, feasible=0.0, violation=viol)

        breedable = set(db._breedable_ids(list(db.programs)))

        # Each group contributes its feasible plus its OWN closest infeasible.
        # Ungrouped, b0 (0.2) would have taken both top-up slots.
        self.assertEqual(breedable, {"a_ok", "a0", "b_ok", "b0"})

    def test_a_group_the_sampler_cannot_see_is_the_bug_this_prevents(self):
        """The pass20 case: every feasible member in one group, the floor met
        by it alone, and a second group that never becomes breedable."""
        ungrouped = _database(
            feasibility_metric="feas", violation_metric="viol", feasibility_min_pool=4
        )
        grouped = self._grouped(feasibility_min_pool=4)
        for db in (ungrouped, grouped):
            for i in range(4):
                self._hold(db, f"a{i}", 0, feasible=1.0, violation=0.0)
            self._hold(db, "b0", 1, feasible=0.0, violation=1.0)

        self.assertNotIn("b0", ungrouped._breedable_ids(list(ungrouped.programs)))
        self.assertIn("b0", grouped._breedable_ids(list(grouped.programs)))
