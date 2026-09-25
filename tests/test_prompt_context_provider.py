"""Optional template providers preserve the default sampler path."""

import random
import unittest
from unittest.mock import patch

from openevolve.config import PromptConfig
from openevolve.prompt.sampler import PromptSampler


def context_provider(context):
    return {"direction": "repair " + str(context["program_metrics"]["score"])}


class TestContextProvider(unittest.TestCase):
    def sampler(self, provider=None):
        config = PromptConfig(
            context_provider=provider, template_variations={"direction": ["random"]}
        )
        sampler = PromptSampler(config)
        sampler.template_manager.templates["full_rewrite_user"] = "{current_program}\n{direction}"
        return sampler

    def test_default_uses_same_random_state_and_prompt(self):
        sampler = self.sampler()
        random.seed(42)
        result = sampler.build_prompt(current_program="source", diff_based_evolution=False)
        state = random.getstate()
        random.seed(42)
        random.choice(["random"])
        self.assertEqual(state, random.getstate())
        self.assertEqual(result["user"], "source\nrandom")

    def test_provider_overrides_variation_without_changing_config(self):
        sampler = self.sampler(__name__ + ":context_provider")
        with patch("openevolve.prompt.sampler.random.choice", side_effect=AssertionError):
            result = sampler.build_prompt(
                current_program="source", program_metrics={"score": 2}, diff_based_evolution=False
            )
        self.assertEqual(result["user"], "source\nrepair 2")
        self.assertEqual(sampler.config.template_variations, {"direction": ["random"]})

    def test_config_round_trip_keeps_provider_importable(self):
        from dataclasses import asdict

        config = self.sampler(__name__ + ":context_provider").config
        sampler = PromptSampler(PromptConfig(**asdict(config)))
        sampler.build_prompt(program_metrics={"score": 1}, diff_based_evolution=False)

    def test_evaluator_template_does_not_call_provider(self):
        sampler = self.sampler(__name__ + ":context_provider")
        sampler.template_manager.templates["evaluation"] = "evaluate {current_program}"
        result = sampler.build_prompt(current_program="x", template_key="evaluation")
        self.assertEqual(result["user"], "evaluate x")

    def test_bad_provider_fails_loudly(self):
        with self.assertRaises(ValueError):
            self.sampler("not-a-provider")

    def test_provider_cannot_replace_contract_fields(self):
        sampler = self.sampler()
        sampler.context_provider = lambda context: {"current_program": "replacement"}
        with self.assertRaises(ValueError):
            sampler.build_prompt(diff_based_evolution=False)

    def test_provider_output_must_be_string_variables(self):
        for output in (None, [], {"direction": 3}, {3: "direction"}):
            with self.subTest(output=output):
                sampler = self.sampler()
                sampler.context_provider = lambda context: output
                with self.assertRaises(ValueError):
                    sampler.build_prompt(diff_based_evolution=False)

    def test_context_includes_parent_artifacts_and_exemplars(self):
        sampler = self.sampler()
        seen = []
        sampler.context_provider = lambda context: seen.append(context) or {"direction": "fixed"}
        sampler.build_prompt(
            program_artifacts={"detail": "failed"},
            top_programs=[{"code": "other", "metrics": {"score": 1}}],
            evolution_round=7,
            diff_based_evolution=False,
        )
        self.assertEqual(seen[0]["program_artifacts"], {"detail": "failed"})
        self.assertEqual(seen[0]["top_programs"][0]["code"], "other")
        self.assertEqual(seen[0]["evolution_round"], 7)
