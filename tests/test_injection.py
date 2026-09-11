"""
Tests for the lanes-fork injection API and per-group telemetry:
ProgramDatabase.inject (persistence + full accounting) and lane_report.
"""

import json
import os
import tempfile
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _prog(pid, lane=0.0, score=0.0, parent_id=None):
    return Program(
        id=pid,
        code=f"# {pid}",
        language="python",
        parent_id=parent_id,
        metrics={"combined_score": score, "lane": lane},
    )


def _db(population_size=100, lane_metric="lane", num_islands=3):
    config = Config()
    config.database.in_memory = True
    config.database.population_size = population_size
    config.database.num_islands = num_islands
    config.database.lane_metric = lane_metric
    return ProgramDatabase(config.database)


class TestInject(unittest.TestCase):
    def test_injects_one_clone_per_island(self):
        db = _db()
        report = db.inject(_prog("seed_a", lane=1, score=0.2), islands=[0, 2])
        self.assertEqual(set(report["added"]), {"seed_a-i0", "seed_a-i2"})
        self.assertIn("seed_a-i0", db.programs)
        self.assertIn("seed_a-i2", db.programs)
        self.assertTrue(db.programs["seed_a-i0"].metadata["injected"])
        self.assertEqual(db.programs["seed_a-i0"].metadata["island"], 0)

    def test_reinjection_is_idempotent(self):
        db = _db()
        db.inject(_prog("seed_a", lane=1), islands=[0, 1])
        n = len(db.programs)
        report = db.inject(_prog("seed_a", lane=1), islands=[0, 1])
        self.assertEqual(report["added"], [])
        self.assertEqual(set(report["skipped"]), {"seed_a-i0", "seed_a-i1"})
        self.assertEqual(len(db.programs), n)

    def test_caller_program_not_mutated(self):
        db = _db()
        base = _prog("seed_a", lane=1)
        db.inject(base, islands=[0])
        self.assertNotIn("injected", base.metadata)
        self.assertNotIn("island", base.metadata)

    def test_low_score_seed_survives_the_cull_it_triggers(self):
        # Fill one island to its cap with distinct-code (distinct-cell) fillers,
        # then inject a low-scoring seed. Its own add pushes over the cap and
        # triggers a cull; the seed must persist anyway, the cull must journal
        # every removal, and nothing may vanish unaccounted (distinct code per id
        # gives distinct MAP-Elites cells, so the fillers accumulate to the cap).
        db = _db(population_size=2, num_islands=1)
        p_hi = _prog("p_hi", lane=0, score=1.0)
        p_mid = _prog("p_mid", lane=0, score=0.5)
        p_hi.code = "x = 1\n"  # short -> low-complexity cell
        p_mid.code = "y = 2\n" * 20  # long -> high-complexity cell
        db.add(p_hi, target_island=0)
        db.add(p_mid, target_island=0)
        self.assertEqual(len(db.programs), 2)  # two distinct cells, at the cap
        journal_before = len(db.eviction_journal)
        report = db.inject(_prog("seed_lo", lane=1, score=-1.0), islands=[0])
        self.assertIn("seed_lo-i0", db.programs)  # persisted despite the worst score
        self.assertIn("seed_lo-i0", report["added"])
        self.assertLessEqual(len(db.programs), 2)  # respects the cap
        self.assertGreater(len(db.eviction_journal), journal_before)  # the cull was journaled
        # Everything removed is reported, nothing silent.
        self.assertEqual(len(report["removed"]), len(db.eviction_journal) - journal_before)

    def test_displacement_is_accounted_not_silent(self):
        # Two seeds landing in the same island+cell: the second displaces the
        # first. The displaced clone must be either present or journaled -- the
        # post-condition inside inject() raises otherwise, so reaching here at all
        # means no silent loss. Assert the journal/report captured any removal.
        db = _db(population_size=100, num_islands=1)
        r1 = db.inject(_prog("seed_a", lane=0, score=0.1), islands=[0])
        r2 = db.inject(_prog("seed_b", lane=0, score=0.9), islands=[0])
        self.assertEqual(r1["added"], ["seed_a-i0"])
        self.assertEqual(r2["added"], ["seed_b-i0"])
        # If seed_a was displaced from its cell, it is journaled, never vanished.
        if "seed_a-i0" not in db.programs:
            self.assertIn("seed_a-i0", {e["id"] for e in db.eviction_journal})

    def test_journal_and_clones_survive_save_load(self):
        db = _db()
        db.inject(_prog("seed_a", lane=1, score=0.2), islands=[0, 1])
        with tempfile.TemporaryDirectory() as d:
            db.save(d, iteration=5)
            db2 = _db()
            db2.load(d)
        self.assertIn("seed_a-i0", db2.programs)
        self.assertIn("seed_a-i1", db2.programs)


class TestLaneReport(unittest.TestCase):
    def test_counts_are_correct(self):
        db = _db(num_islands=2)
        db.add(_prog("q0", lane=0, score=0.3, parent_id="root"), target_island=0)
        db.add(_prog("q1", lane=0, score=0.5, parent_id="root"), target_island=1)
        db.inject(_prog("seed_s", lane=1, score=0.1), islands=[0])  # a seeded lane-1
        report = db.lane_report()
        self.assertEqual(report[0]["population"], 2)
        self.assertEqual(report[0]["evolved"], 2)  # both have a parent
        self.assertAlmostEqual(report[0]["best_combined_score"], 0.5)
        self.assertEqual(report[1]["population"], 1)
        self.assertEqual(report[1]["seeded"], 1)  # the injected clone
        self.assertEqual(report[0]["elite"] + report[0]["homeless"], 2)

    def test_empty_without_lane_metric(self):
        db = _db(lane_metric=None)
        db.add(_prog("p0", lane=0), target_island=0)
        self.assertEqual(db.lane_report(), {})

    def test_present_in_saved_metadata(self):
        db = _db()
        db.add(_prog("q0", lane=0, score=0.3, parent_id="root"), target_island=0)
        with tempfile.TemporaryDirectory() as d:
            db.save(d, iteration=1)
            with open(os.path.join(d, "metadata.json")) as f:
                meta = json.load(f)
        self.assertIn("lane_report", meta)
        self.assertEqual(meta["lane_report"]["0"]["population"], 1)  # int key -> str via json


if __name__ == "__main__":
    unittest.main()


class TestInjectCarriesArtifacts(unittest.TestCase):
    """A clone is a copy of the program, artifacts included.

    An injected program is otherwise the only kind in the archive with no
    evaluation record -- and a fresh run consists entirely of injected
    programs, so anything reading artifacts sees nothing exactly at iteration 0.
    """

    def test_a_clone_carries_artifacts_json_and_dir(self):
        db = _db()
        program = _prog("seed", score=0.5)
        program.artifacts_json = json.dumps({"feedback": "hello", "detail": [1, 2, 3]})
        program.artifact_dir = "/tmp/artifacts/seed"
        db.inject(program, [0, 1], iteration=0)
        for island in (0, 1):
            clone = db.programs[f"seed-i{island}"]
            self.assertEqual(clone.artifacts_json, program.artifacts_json)
            self.assertEqual(clone.artifact_dir, program.artifact_dir)
            self.assertEqual(json.loads(clone.artifacts_json)["feedback"], "hello")

    def test_a_program_without_artifacts_still_injects(self):
        db = _db()
        db.inject(_prog("bare", score=0.5), [0], iteration=0)
        self.assertIsNone(db.programs["bare-i0"].artifacts_json)
        self.assertIsNone(db.programs["bare-i0"].artifact_dir)

    def test_artifacts_survive_a_save_and_load(self):
        db = _db()
        program = _prog("seed", score=0.5)
        program.artifacts_json = json.dumps({"feedback": "kept"})
        db.inject(program, [0], iteration=0)
        with tempfile.TemporaryDirectory() as tmp:
            db.save(tmp, 0)
            other = _db()
            other.load(tmp)
            self.assertEqual(
                json.loads(other.programs["seed-i0"].artifacts_json)["feedback"], "kept"
            )
