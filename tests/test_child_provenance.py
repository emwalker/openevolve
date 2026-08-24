"""
The evolved child records which programs were offered as inspiration
(metadata["inspiration_ids"]), so cross-group exposure can be attributed later.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import openevolve.process_parallel as pp
from openevolve.config import Config
from openevolve.database import Program


def _prog(pid, island=0):
    return Program(id=pid, code=f"# {pid}\n", language="python", metadata={"island": island})


class TestChildInspirationProvenance(unittest.TestCase):
    def test_worker_records_inspiration_ids(self):
        parent = _prog("parent")
        insp = _prog("insp1")
        db_snapshot = {
            "programs": {"parent": parent.to_dict(), "insp1": insp.to_dict()},
            "artifacts": {},
            "current_island": 0,
            "islands": [["parent", "insp1"]],
            "feature_dimensions": [],
            "sampling_island": 0,
        }

        config = Config()
        config.diff_based_evolution = False
        config.max_code_length = 100000

        sampler = MagicMock()
        sampler.build_prompt.return_value = {"system": "s", "user": "u"}
        llm = MagicMock()
        llm.generate_with_context = AsyncMock(return_value="<response>")
        evaluator = MagicMock()
        evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 0.5})
        evaluator.get_pending_artifacts.return_value = None

        with (
            patch.multiple(
                pp,
                create=True,
                _worker_config=config,
                _worker_prompt_sampler=sampler,
                _worker_llm_ensemble=llm,
                _worker_evaluator=evaluator,
            ),
            patch("openevolve.utils.code_utils.parse_full_rewrite", return_value="y = 2\n"),
        ):
            result = pp._run_iteration_worker(
                iteration=7,
                db_snapshot=db_snapshot,
                parent_id="parent",
                inspiration_ids=["insp1"],
            )

        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.child_program_dict["metadata"]["inspiration_ids"], ["insp1"])


if __name__ == "__main__":
    unittest.main()
