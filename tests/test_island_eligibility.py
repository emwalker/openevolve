"""Island exploration applies constraint eligibility once."""

import unittest
from unittest.mock import patch

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class TestIslandEligibility(unittest.TestCase):
    def database(self, grouped=True):
        config = DatabaseConfig(
            in_memory=True,
            num_islands=1,
            feasibility_metric="valid",
            violation_metric="violation",
            feasibility_min_pool=4,
            lane_metric="group" if grouped else None,
            lane_sampling_gamma=1.0 if grouped else None,
        )
        db = ProgramDatabase(config)
        for i in range(8):
            program = Program(
                id=str(i),
                code=f"value = {i}",
                metrics={
                    "valid": 0.0,
                    "violation": float(i),
                    "group": 0.0,
                    "combined_score": float(-i),
                },
            )
            db.programs[program.id] = program
        db.islands[0] = set(db.programs)
        return db

    def test_grouped_exploration_keeps_closest_half_not_quarter(self):
        db = self.database()
        with patch.object(
            db, "_group_weighted_choice", side_effect=lambda ids: db.programs[ids[0]]
        ) as draw:
            db._sample_from_island_random(0)
        self.assertEqual(set(draw.call_args.args[0]), {"0", "1", "2", "3"})

    def test_ungrouped_exploration_keeps_closest_half(self):
        db = self.database(grouped=False)
        with patch("openevolve.database.random.choice", side_effect=lambda ids: ids[0]) as draw:
            db._sample_from_island_random(0)
        self.assertEqual(set(draw.call_args.args[0]), {"0", "1", "2", "3"})

    def test_feasible_pool_still_tops_up_to_its_minimum(self):
        db = self.database()
        db.programs["7"].metrics["valid"] = 1.0
        with patch.object(
            db, "_group_weighted_choice", side_effect=lambda ids: db.programs[ids[0]]
        ) as draw:
            db._sample_from_island_random(0)
        self.assertEqual(set(draw.call_args.args[0]), {"7", "0", "1", "2"})

    def test_unconfigured_exploration_still_removes_stale_ids(self):
        db = self.database(grouped=False)
        db.feasibility_metric = None
        db.islands[0].add("stale")
        with patch("openevolve.database.random.choice", side_effect=lambda ids: ids[0]) as draw:
            db._sample_from_island_random(0)
        self.assertEqual(set(draw.call_args.args[0]), set(db.programs))
