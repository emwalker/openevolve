"""
Tests for the lanes-fork group-aware sampling additions to ProgramDatabase:
group-balanced parent weighting, within-group rank weighting, group-scoped
inspirations, and the shared lane_group_of helper the workers reuse. All are
inert unless lane_metric is configured (defaults reproduce upstream behavior).
"""

import unittest
from unittest import mock

import openevolve.database as oe_db
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase, lane_group_of


def _prog(pid: str, lane: float, score: float = 0.0, island: int = 0) -> Program:
    return Program(
        id=pid,
        code=f"# {pid}",
        language="python",
        metrics={"combined_score": score, "lane": lane},
        metadata={"island": island},
    )


def _db(gamma=None, w=0.0, p=1.0, scope=False, cross=1, lane_metric="lane"):
    config = Config()
    config.database.in_memory = True
    config.database.lane_metric = lane_metric
    config.database.lane_sampling_gamma = gamma
    config.database.lane_rank_weight = w
    config.database.lane_rank_power = p
    config.database.lane_prompt_scope = scope
    config.database.lane_cross_inspirations = cross
    return ProgramDatabase(config.database)


def _group_totals(weights, programs):
    totals = {}
    for weight, program in zip(weights, programs):
        lane = program.metrics["lane"]
        totals[lane] = totals.get(lane, 0.0) + weight
    return totals


class TestGroupWeights(unittest.TestCase):
    """_group_weights: each group's aggregate share is n**(1-gamma), split within
    the group by score rank (w=0 -> uniform)."""

    def test_group_totals_match_n_pow_one_minus_gamma(self):
        for w in (0.0, 0.75):
            db = _db(gamma=1.0, w=w)
            programs = [
                _prog("a", 0, 0.01),
                _prog("b", 0, 0.02),
                _prog("c", 0, 0.03),
                _prog("d", 1, 0.05),
            ]
            for gamma in (0.0, 1.0, 2.0, 3.0):
                db.lane_sampling_gamma = gamma
                totals = _group_totals(db._group_weights(programs), programs)
                self.assertAlmostEqual(totals[0], 3 ** (1.0 - gamma))
                self.assertAlmostEqual(totals[1], 1 ** (1.0 - gamma))

    def test_w_zero_is_uniform_within_group(self):
        db = _db(gamma=3.0, w=0.0)
        programs = [_prog("a", 0, 0.9), _prog("b", 0, 0.1), _prog("c", 0, 0.5)]
        weights = db._group_weights(programs)
        self.assertAlmostEqual(weights[0], 3.0**-3.0)
        self.assertEqual(len(set(round(x, 12) for x in weights)), 1)  # uniform regardless of score

    def test_balanced_groups_get_equal_totals_for_any_gamma(self):
        programs = [_prog(f"p{i}", lane) for i, lane in enumerate([0, 0, 1, 1, 2, 2])]
        for gamma in (0.0, 1.0, 2.0):
            db = _db(gamma=gamma, w=0.75)
            totals = _group_totals(db._group_weights(programs), programs)
            self.assertAlmostEqual(totals[0], totals[1])
            self.assertAlmostEqual(totals[1], totals[2])

    def test_underpopulated_group_gets_higher_per_program_weight(self):
        db = _db(gamma=1.0, w=0.0)
        programs = [_prog("a", 0), _prog("b", 0), _prog("c", 0), _prog("d", 1)]
        weights = db._group_weights(programs)
        self.assertGreater(weights[-1], weights[0])  # lone group-1 outweighs any group-0

    def test_higher_score_gets_strictly_higher_weight_within_group(self):
        db = _db(gamma=3.0, w=0.75, p=3.0)
        programs = [_prog(x, 0, s) for x, s in [("a", 0.01), ("b", 0.05), ("c", 0.02), ("d", 0.09)]]
        weights = db._group_weights(programs)
        self.assertTrue(weights[3] > weights[1] > weights[2] > weights[0])

    def test_worst_program_keeps_a_positive_floor(self):
        db = _db(gamma=3.0, w=0.75, p=3.0)
        programs = [_prog(f"p{i}", 0, i * 0.001) for i in range(30)]
        self.assertGreater(min(db._group_weights(programs)), 0.0)

    def test_missing_score_ranks_last_but_keeps_the_floor(self):
        db = _db(gamma=3.0, w=0.75, p=3.0)
        programs = [_prog("a", 0, 0.05), _prog("b", 0, 0.02), _prog("c", 0, 0.0)]
        programs[2].metrics["combined_score"] = None  # no score -> ranks bottom
        weights = db._group_weights(programs)
        self.assertEqual(weights[2], min(weights))
        self.assertGreater(weights[2], 0.0)

    def test_group_weighted_choice_passes_group_weights(self):
        db = _db(gamma=1.0, w=0.75)
        programs = [_prog(x, 0, s) for x, s in [("a", 0.01), ("b", 0.02), ("c", 0.03)]]
        programs.append(_prog("d", 1, 0.05))
        for program in programs:
            db.programs[program.id] = program
        seen = {}

        def fake_choices(population, weights, k):
            seen["weights"] = list(weights)
            return [population[0]]

        with mock.patch.object(oe_db.random, "choices", fake_choices):
            db._group_weighted_choice([p.id for p in programs])
        self.assertEqual(seen["weights"], db._group_weights(programs))


class TestGroupScopedInspirations(unittest.TestCase):
    """_sample_inspirations_scoped: best same-group members plus cross-group tail."""

    def _seed(self, db, programs):
        db.programs = {p.id: p for p in programs}
        db.islands = [{p.id for p in programs}]
        db.current_island = 0

    def test_best_same_group_plus_one_cross_group(self):
        db = _db(scope=True, cross=1)
        parent = _prog("parent", 1, 0.01)
        progs = [parent, _prog("hi", 1, 0.05), _prog("lo", 1, -0.02), _prog("other", 0, 0.10)]
        self._seed(db, progs)
        chosen = db._sample_inspirations_scoped(parent, n=2, island_id=0)
        self.assertEqual([p.id for p in chosen], ["hi", "other"])

    def test_cross_zero_stays_within_group(self):
        db = _db(scope=True, cross=0)
        parent = _prog("parent", 1, 0.01)
        progs = [parent, _prog("hi", 1, 0.05), _prog("mid", 1, 0.03), _prog("other", 0, 0.10)]
        self._seed(db, progs)
        chosen = db._sample_inspirations_scoped(parent, n=2, island_id=0)
        self.assertEqual([p.id for p in chosen], ["hi", "mid"])

    def test_empty_group_falls_back_to_upstream(self):
        db = _db(scope=True, cross=1)
        parent = _prog("parent", 3, 0.0)  # lone group-3 member on this island
        self._seed(db, [parent, _prog("other", 0, 0.10)])
        self.assertIsNone(db._sample_inspirations_scoped(parent, n=2, island_id=0))

    def test_sample_inspirations_delegates_when_scope_on(self):
        db = _db(scope=True, cross=1)
        parent = _prog("parent", 1, 0.01)
        progs = [parent, _prog("hi", 1, 0.05), _prog("other", 0, 0.10)]
        self._seed(db, progs)
        db.island_best_programs = [None]
        chosen = db._sample_inspirations(parent, n=2, island_id=0)
        self.assertEqual([p.id for p in chosen], ["hi", "other"])


class TestLaneGroupHelper(unittest.TestCase):
    """lane_group_of: the domain-free grouping the workers reuse."""

    def test_rounds_to_int_bucket(self):
        self.assertEqual(lane_group_of(_prog("a", 2.0), "lane"), 2)
        self.assertEqual(lane_group_of(_prog("a", 2.4), "lane"), 2)

    def test_missing_metric_is_minus_one(self):
        self.assertEqual(lane_group_of(_prog("a", 0.0, 0.0), "absent"), -1)

    def test_none_metric_means_upstream(self):
        self.assertIsNone(lane_group_of(_prog("a", 0.0), None))

    def test_filter_keeps_only_parent_group(self):
        programs = [_prog("a", 0), _prog("b", 1), _prog("c", 0), _prog("d", 1)]
        parent_group = lane_group_of(programs[0], "lane")
        kept = [p for p in programs if lane_group_of(p, "lane") == parent_group]
        self.assertEqual([p.id for p in kept], ["a", "c"])


class TestDefaultsAreUpstream(unittest.TestCase):
    """With the new fields unset, group-aware paths stay inert."""

    def test_defaults_off(self):
        db = _db(gamma=None, scope=False, lane_metric=None)
        self.assertIsNone(db.lane_sampling_gamma)
        self.assertFalse(db.lane_prompt_scope)

    def test_scope_without_lane_metric_is_disabled(self):
        db = _db(gamma=2.0, scope=True, lane_metric=None)
        self.assertIsNone(db.lane_sampling_gamma)  # cleared with a warning
        self.assertFalse(db.lane_prompt_scope)


if __name__ == "__main__":
    unittest.main()
