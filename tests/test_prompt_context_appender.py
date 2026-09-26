"""Extra context leaves selected guidance and template formatting intact."""

import random
import unittest
from dataclasses import asdict

from openevolve.config import PromptConfig
from openevolve.prompt.sampler import PromptSampler


def append_context(context):
    return "\n\nEvidence: " + context["recent_attempts"][0]["code"]


class TestContextAppender(unittest.TestCase):
    def sampler(self, enabled=True):
        config = PromptConfig(
            context_appender=__name__ + ":append_context" if enabled else None,
            template_variations={"direction": ["explore", "repair"]},
        )
        sampler = PromptSampler(config)
        sampler.template_manager.templates["full_rewrite_user"] = "{direction}\n{current_program}"
        return sampler

    def build(self, sampler):
        random.seed(42)
        prompt = sampler.build_prompt(
            current_program="parent",
            diff_based_evolution=False,
            recent_attempts=[{"code": "child = {literal}"}],
        )
        return prompt, random.getstate()

    def test_appends_literal_evidence_preserving_guidance_and_rng(self):
        before, before_rng = self.build(self.sampler(False))
        after, after_rng = self.build(self.sampler())
        self.assertEqual(after_rng, before_rng)
        self.assertEqual(after["system"], before["system"])
        self.assertEqual(after["user"], before["user"] + "\n\nEvidence: child = {literal}")

    def test_empty_context_preserves_prompt_exactly(self):
        sampler = self.sampler()
        sampler.context_appender = lambda context: ""
        self.assertEqual(self.build(sampler), self.build(self.sampler(False)))

    def test_template_provider_and_appender_compose(self):
        sampler = self.sampler()
        sampler.context_provider = lambda context: {"direction": "targeted"}
        prompt, _ = self.build(sampler)
        self.assertEqual(prompt["user"], "targeted\nparent\n\nEvidence: child = {literal}")

    def test_evaluation_does_not_call_appender(self):
        sampler = self.sampler()
        sampler.context_appender = lambda context: self.fail("called during evaluation")
        sampler.template_manager.templates["evaluation"] = "evaluate {current_program}"
        prompt = sampler.build_prompt(current_program="parent", template_key="evaluation")
        self.assertEqual(prompt["user"], "evaluate parent")

    def test_appender_survives_config_round_trip(self):
        sampler = self.sampler()
        restored = PromptSampler(PromptConfig(**asdict(sampler.config)))
        restored.template_manager.templates["full_rewrite_user"] = "{direction}\n{current_program}"
        self.assertEqual(self.build(sampler), self.build(restored))

    def test_invalid_output_is_refused(self):
        for output in (None, {}, 3):
            with self.subTest(output=output):
                sampler = self.sampler()
                sampler.context_appender = lambda context: output
                with self.assertRaisesRegex(ValueError, "context_appender.*string"):
                    self.build(sampler)

    def test_invalid_appender_is_refused(self):
        with self.assertRaisesRegex(ValueError, "context_appender"):
            PromptSampler(PromptConfig(context_appender="invalid"))
