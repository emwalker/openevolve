"""One exemplar per distinct behaviour, and the child metadata that identifies one.

The generator copies what it is shown. A population holding many copies of one
program spends every prompt slot on that one program unless the lists it is
shown are deduplicated by what a program *is* rather than by its text.
"""

import time
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase, dedup_exemplars
from openevolve.process_parallel import _exemplar_slices, _metadata_from_artifacts


def _program(pid, *, score=0.5, behaviour=None):
    metadata = {} if behaviour is None else {"behaviour": behaviour}
    return Program(
        id=pid,
        code=f"# {pid}",
        metrics={"combined_score": score},
        metadata=metadata,
        timestamp=time.time(),
    )


def _database(**overrides):
    config = Config().database
    config.in_memory = True
    config.num_islands = 1
    for key, value in overrides.items():
        setattr(config, key, value)
    return ProgramDatabase(config)


class TestDedupExemplars(unittest.TestCase):
    def test_one_member_per_behaviour_in_rank_order(self):
        programs = [
            _program("a", behaviour="x"),
            _program("b", behaviour="y"),
            _program("c", behaviour="x"),
            _program("d", behaviour="y"),
            _program("e", behaviour="z"),
        ]
        kept = [p.id for p in dedup_exemplars(programs, "behaviour")]
        self.assertEqual(kept, ["a", "b", "e"])

    def test_an_unidentified_program_is_never_collapsed(self):
        """Two programs the evaluator could not identify are not thereby the
        same program."""
        programs = [_program("a", behaviour="x"), _program("b"), _program("c")]
        self.assertEqual([p.id for p in dedup_exemplars(programs, "behaviour")], ["a", "b", "c"])

    def test_unconfigured_is_the_identity(self):
        programs = [_program("a", behaviour="x"), _program("b", behaviour="x")]
        self.assertEqual(dedup_exemplars(programs, None), programs)

    def test_get_top_programs_never_shows_a_behaviour_twice(self):
        db = _database(dedup_key="behaviour")
        db.add(_program("best", score=0.9, behaviour="x"))
        db.add(_program("twin", score=0.8, behaviour="x"))
        db.add(_program("other", score=0.7, behaviour="y"))
        self.assertEqual([p.id for p in db.get_top_programs(n=3)], ["best", "other"])

    def test_an_explicit_metric_is_not_deduplicated(self):
        """`get_top_programs(metric=...)` answers a question about that metric
        over every program, not about the exemplar list."""
        db = _database(dedup_key="behaviour")
        db.add(_program("best", score=0.9, behaviour="x"))
        db.add(_program("twin", score=0.8, behaviour="x"))
        ranked = db.get_top_programs(n=3, metric="combined_score")
        self.assertEqual([p.id for p in ranked], ["best", "twin"])


class TestMetadataFromArtifacts(unittest.TestCase):
    def test_a_configured_key_present_in_artifacts_lands_in_metadata(self):
        got = _metadata_from_artifacts(["behaviour_sha1"], {"behaviour_sha1": "abc", "other": 1})
        self.assertEqual(got, {"behaviour_sha1": "abc"})

    def test_an_absent_key_sets_nothing(self):
        self.assertEqual(_metadata_from_artifacts(["behaviour_sha1"], {"other": 1}), {})
        self.assertEqual(_metadata_from_artifacts(["behaviour_sha1"], None), {})

    def test_unconfigured_copies_nothing(self):
        self.assertEqual(_metadata_from_artifacts([], {"behaviour_sha1": "abc"}), {})

    def test_a_string_survives(self):
        """The whole point: metrics are coerced to float, metadata is not."""
        got = _metadata_from_artifacts(["k"], {"k": "a" * 40})
        self.assertIsInstance(got["k"], str)


class TestExemplarSlices(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.prompt.num_top_programs = 3
        self.config.prompt.num_diverse_programs = 2
        self.ranked = [_program(str(i), score=1.0 - i / 10) for i in range(10)]

    def test_upstream_takes_both_slices_from_the_head(self):
        best, prompt = _exemplar_slices(self.config, self.ranked)
        self.assertEqual([p.id for p in best], ["0", "1", "2"])
        self.assertEqual([p.id for p in prompt], ["0", "1", "2", "3", "4"])

    def test_island_wide_draws_the_diverse_slots_from_below_the_top(self):
        self.config.prompt.diverse_from_island = True
        best, prompt = _exemplar_slices(self.config, self.ranked)
        self.assertEqual([p.id for p in best], ["0", "1", "2"])
        self.assertEqual([p.id for p in prompt[:3]], ["0", "1", "2"])
        self.assertEqual(len(prompt), 5)
        for p in prompt[3:]:
            self.assertNotIn(p.id, {"0", "1", "2"})

    def test_island_wide_draws_vary(self):
        self.config.prompt.diverse_from_island = True
        seen = set()
        for _ in range(40):
            _, prompt = _exemplar_slices(self.config, self.ranked)
            seen.add(tuple(sorted(p.id for p in prompt[3:])))
        self.assertGreater(len(seen), 1, "the diverse slots never varied")

    def test_fewer_remaining_than_asked_for_is_not_an_error(self):
        self.config.prompt.diverse_from_island = True
        best, prompt = _exemplar_slices(self.config, self.ranked[:4])
        self.assertEqual(len(best), 3)
        self.assertEqual(len(prompt), 4)

    def test_a_group_smaller_than_the_top_slice(self):
        self.config.prompt.diverse_from_island = True
        best, prompt = _exemplar_slices(self.config, self.ranked[:2])
        self.assertEqual(len(best), 2)
        self.assertEqual(len(prompt), 2)


if __name__ == "__main__":
    unittest.main()
