"""The exemplar ordering and the flag the prompt renders from it.

`rank_exemplars` is exercised through the database in `test_feasibility.py`;
here it is tested as the free function the evaluation worker calls, which holds
no database and reads its settings off the config.
"""

import time
import unittest

from openevolve.database import Program, rank_exemplars
from openevolve.process_parallel import _exemplar_dicts


def _program(pid, *, score, feasible=None, violation=None):
    metrics = {"combined_score": score}
    if feasible is not None:
        metrics["feas"] = feasible
    if violation is not None:
        metrics["viol"] = violation
    return Program(id=pid, code=f"# {pid}", metrics=metrics, timestamp=time.time())


class TestRankExemplars(unittest.TestCase):
    def _order(self, programs, metric="feas"):
        return [p.id for p in rank_exemplars(programs, metric, "viol", None)]

    def test_feasible_first_whatever_the_scores(self):
        programs = [
            _program("bad", score=0.9, feasible=0.0, violation=1.0),
            _program("good", score=0.1, feasible=1.0, violation=0.0),
        ]
        self.assertEqual(self._order(programs), ["good", "bad"])

    def test_infeasible_by_violation_then_fitness(self):
        programs = [
            _program("far", score=0.9, feasible=0.0, violation=5.0),
            _program("near_lo", score=0.2, feasible=0.0, violation=0.5),
            _program("near_hi", score=0.7, feasible=0.0, violation=0.5),
        ]
        self.assertEqual(self._order(programs), ["near_hi", "near_lo", "far"])

    def test_unconfigured_is_the_plain_fitness_sort(self):
        programs = [
            _program("lo", score=0.1, feasible=0.0, violation=9.0),
            _program("hi", score=0.9, feasible=0.0, violation=9.0),
        ]
        self.assertEqual(self._order(programs, metric=None), ["hi", "lo"])

    def test_an_unmeasured_program_is_treated_as_feasible(self):
        """Missing is not in breach -- `is_feasible`'s rule, and the ordering
        must not invent a violation for a program nobody measured."""
        programs = [
            _program("bad", score=0.9, feasible=0.0, violation=1.0),
            _program("unmeasured", score=0.5),
        ]
        self.assertEqual(self._order(programs), ["unmeasured", "bad"])

    def test_the_order_is_stable_for_equal_keys(self):
        a = _program("a", score=0.5, feasible=1.0, violation=0.0)
        b = _program("b", score=0.5, feasible=1.0, violation=0.0)
        self.assertEqual(self._order([a, b]), ["a", "b"])
        self.assertEqual(self._order([b, a]), ["b", "a"])


class TestExemplarDicts(unittest.TestCase):
    def test_only_the_infeasible_program_is_stamped(self):
        dicts = _exemplar_dicts(
            [
                _program("good", score=0.1, feasible=1.0, violation=0.0),
                _program("bad", score=0.9, feasible=0.0, violation=1.0),
            ],
            "feas",
        )
        self.assertNotIn("infeasible", dicts[0])
        self.assertTrue(dicts[1]["infeasible"])

    def test_nothing_is_stamped_when_unconfigured(self):
        dicts = _exemplar_dicts([_program("bad", score=0.9, feasible=0.0)], None)
        self.assertNotIn("infeasible", dicts[0])

    def test_the_dict_is_still_the_program_dict(self):
        program = _program("good", score=0.1, feasible=1.0, violation=0.0)
        self.assertEqual(_exemplar_dicts([program], "feas")[0]["code"], program.code)


if __name__ == "__main__":
    unittest.main()
