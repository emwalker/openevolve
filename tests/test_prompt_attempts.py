"""Recent attempts are parent-specific evidence, independent of ranking."""

import json
import random
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openevolve import process_parallel
from openevolve.config import Config, PromptConfig
from openevolve.database import Program
from openevolve.prompt.attempts import recent_attempts
from openevolve.prompt.sampler import PromptSampler


def program(pid, code, parent=None, iteration=0, group=1):
    return Program(
        id=pid,
        code=code,
        parent_id=parent,
        iteration_found=iteration,
        timestamp=float(iteration),
        metrics={"group": group, "score": -iteration},
    )


class TestRecentAttempts(unittest.TestCase):
    def setUp(self):
        self.parent = program("parent", "original")
        self.copy = program("other-island", "original")
        self.child = program("child", "changed", self.copy.id, 2)
        self.child.artifacts_json = json.dumps({"feedback": "failed check"})
        self.programs = {p.id: p for p in (self.parent, self.copy, self.child)}

    def test_cross_island_source_copies_share_attempts_and_feedback(self):
        result = recent_attempts(self.programs, self.parent, 3, "group")
        self.assertEqual([p["id"] for p in result], [self.child.id])
        self.assertEqual(result[0]["artifacts"], {"feedback": "failed check"})
        self.assertEqual(result[0]["code"], "changed")
        self.assertNotIn("prompts", result[0])

    def test_excludes_unrelated_sources_and_other_groups(self):
        other = program("unrelated", "different")
        foreign = program("foreign", "original", group=2)
        for p in (
            other,
            foreign,
            program("x", "x", other.id, 5),
            program("y", "y", foreign.id, 6, group=2),
            program("z", "z", self.parent.id, 7, group=2),
        ):
            self.programs[p.id] = p
        self.assertEqual(len(recent_attempts(self.programs, self.parent, 3, "group")), 1)

    def test_newest_distinct_sources_first_without_consuming_rng(self):
        for p in (
            program("duplicate", "changed", self.parent.id, 3),
            program("new", "new", self.parent.id, 4),
        ):
            self.programs[p.id] = p
        state = random.getstate()
        result = recent_attempts(self.programs, self.parent, 2, "group")
        self.assertEqual([p["id"] for p in result], ["new", "duplicate"])
        self.assertEqual(state, random.getstate())

    def test_disabled_does_not_read_population(self):
        self.assertEqual(recent_attempts(None, self.parent, 0, "group"), [])
        self.assertEqual(PromptConfig().num_recent_attempts, 0)

    def test_attempts_reach_provider_without_replacing_exemplars(self):
        sampler = PromptSampler(PromptConfig())
        seen = []
        sampler.context_provider = lambda context: seen.append(context) or {}
        attempts = recent_attempts(self.programs, self.parent, 3, "group")
        sampler.build_prompt(current_program="original", recent_attempts=attempts)
        self.assertEqual(seen[0]["recent_attempts"], attempts)
        self.assertFalse(seen[0]["top_programs"])

    def test_worker_supplies_attempts_even_when_they_are_not_top_exemplars(self):
        config = Config()
        config.database.lane_metric = "group"
        config.prompt.num_recent_attempts = 3
        config.prompt.num_top_programs = 1
        config.prompt.num_diverse_programs = 0
        sampler = PromptSampler(PromptConfig(**asdict(config.prompt)))
        seen = []
        sampler.context_provider = lambda context: seen.append(context) or {}
        snapshot = {
            "programs": {pid: p.to_dict() for pid, p in self.programs.items()},
            "artifacts": {},
            "islands": [[self.parent.id], [self.copy.id, self.child.id]],
            "current_island": 0,
        }
        with (
            patch.object(process_parallel, "_lazy_init_worker_components"),
            patch.multiple(
                process_parallel,
                _worker_config=config,
                _worker_prompt_sampler=sampler,
                _worker_llm_ensemble=SimpleNamespace(
                    generate_with_context=AsyncMock(return_value=None)
                ),
                create=True,
            ),
        ):
            result = process_parallel._run_iteration_worker(3, snapshot, self.parent.id, [])
        self.assertEqual(result.error, "LLM returned None response")
        self.assertEqual([p["id"] for p in seen[0]["top_programs"]], [self.parent.id])
        self.assertEqual(seen[0]["recent_attempts"][0]["artifacts"]["feedback"], "failed check")
