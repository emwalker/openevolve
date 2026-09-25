"""Nominated parent trials are finite, cross-island, and checkpointed."""

import json
import random
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


def database(values=("trial",)):
    config = DatabaseConfig(
        in_memory=True,
        num_islands=2,
        feasibility_metric="feasible",
        violation_metric="violation",
        exploration_ratio=1.0,
        exploitation_ratio=0.0,
        parent_trial_key="label" if values else None,
        parent_trial_values=list(values),
    )
    db = ProgramDatabase(config)
    for island in range(2):
        for label, feasible in (("best", 1.0), ("trial", 0.0)):
            p = Program(
                id=f"{label}-{island}",
                code=label,
                metrics={
                    "combined_score": feasible,
                    "feasible": feasible,
                    "violation": 1 - feasible,
                },
                metadata={"label": label},
            )
            db.add(p, target_island=island)
    return db


class TestParentTrials(unittest.TestCase):
    def test_nomination_bypasses_breeding_filter_once_not_acceptance(self):
        db = database()
        parent, _ = db.sample_from_island(0, 0)
        self.assertEqual(parent.id, "trial-0")
        self.assertEqual(parent.metrics["feasible"], 0)
        self.assertEqual(db.get_best_program().metrics["feasible"], 1)
        self.assertEqual(db.sample_from_island(1, 0)[0].id, "best-1")
        report = db.parent_trial_report()
        self.assertEqual(len(report["draws"]), 1)
        self.assertFalse(report["draws"][0]["normally_eligible"])

    def test_sample_and_island_sample_share_one_quota(self):
        db = database()
        self.assertEqual(db.sample(0)[0].id, "trial-0")
        self.assertEqual(db.sample_from_island(1, 0)[0].id, "best-1")

    def test_resume_does_not_repeat_trial(self):
        db = database()
        db.sample_from_island(0, 0)
        with tempfile.TemporaryDirectory() as path:
            db.save(path)
            restored = ProgramDatabase(db.config)
            restored.load(path)
            self.assertEqual(restored.parent_trial_report(), db.parent_trial_report())
            self.assertEqual(restored.sample_from_island(1, 0)[0].id, "best-1")

    def test_missing_local_copy_remains_pending_for_another_island(self):
        db = database()
        db.islands[0].remove("trial-0")
        self.assertEqual(db.sample_from_island(0, 0)[0].id, "best-0")
        self.assertEqual(db.sample_from_island(1, 0)[0].id, "trial-1")

    def test_unknown_or_ambiguous_nomination_is_refused(self):
        db = database(("trial", "missing"))
        with self.assertRaisesRegex(ValueError, "missing"):
            db.sample_from_island(0, 0)
        self.assertEqual(db.parent_trial_report()["draws"], [])
        db = database()
        db.programs["trial-1"].code = "different source"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            db.sample_from_island(0, 0)

    def test_same_label_in_different_groups_is_ambiguous(self):
        db = database()
        db.lane_metric = "group"
        db.programs["trial-0"].metrics["group"] = 1
        db.programs["trial-1"].metrics["group"] = 2
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            db.sample_from_island(0, 0)

    def test_concurrent_islands_reserve_only_one_trial(self):
        db = database()
        with ThreadPoolExecutor(max_workers=2) as pool:
            parents = list(pool.map(lambda island: db.sample_from_island(island, 0)[0], (0, 1)))
        self.assertEqual(sum(p.metadata["label"] == "trial" for p in parents), 1)

    def test_temporarily_disabling_trials_preserves_consumed_quota(self):
        db = database()
        db.sample_from_island(0, 0)
        with tempfile.TemporaryDirectory() as path:
            db.save(path)
            disabled = ProgramDatabase(database(()).config)
            disabled.load(path)
            disabled.save(path)
            resumed = ProgramDatabase(db.config)
            resumed.load(path)
            self.assertEqual(resumed.sample_from_island(1, 0)[0].id, "best-1")

    def test_changed_trial_list_on_resume_is_refused(self):
        db = database()
        db.sample_from_island(0, 0)
        with tempfile.TemporaryDirectory() as path:
            db.save(path)
            config = database(("best",)).config
            with self.assertRaisesRegex(ValueError, "trial configuration"):
                ProgramDatabase(config).load(path)

    def test_missing_nomination_is_refused_while_loading_initial_checkpoint(self):
        db = database(())
        with tempfile.TemporaryDirectory() as path:
            db.save(path)
            with self.assertRaisesRegex(ValueError, "missing"):
                ProgramDatabase(database(("missing",)).config).load(path)

    def test_disabled_checkpoint_has_no_trial_state(self):
        db = database(())
        with tempfile.TemporaryDirectory() as path:
            db.save(path)
            with open(path + "/metadata.json") as handle:
                self.assertNotIn("parent_trials", json.load(handle))
        state = random.getstate()
        self.assertIsNone(db.parent_trial_report())
        self.assertEqual(state, random.getstate())

    def test_invalid_trial_config_is_refused(self):
        for key, values in ((None, ["x"]), ("label", []), ("label", ["x", "x"]), ("label", [""])):
            with self.subTest(key=key, values=values), self.assertRaises(ValueError):
                ProgramDatabase(DatabaseConfig(parent_trial_key=key, parent_trial_values=values))
