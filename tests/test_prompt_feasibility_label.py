"""An infeasible program reaches the prompt labelled, or its score lies.

The prompt shows a near-miss for its ideas; the score beside it is the one
number about it not worth copying, so the header has to say which it is
offering.
"""

import unittest

from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler

LABEL = "INFEASIBLE"


class TestFeasibilityLabel(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.prompt.num_top_programs = 2
        config.prompt.num_diverse_programs = 0
        config.prompt.include_artifacts = False
        self.sampler = PromptSampler(config.prompt)

    def _prog(self, pid, score, infeasible=False):
        program = {
            "id": pid,
            "code": f"def {pid}(): return {score}",
            "metrics": {"combined_score": score},
        }
        if infeasible:
            program["infeasible"] = True
        return program

    def _build(self, top=None, inspirations=None, previous=None):
        return self.sampler.build_prompt(
            current_program="def cur(): pass",
            parent_program="def cur(): pass",
            program_metrics={"combined_score": 0.5},
            top_programs=top or [],
            inspirations=inspirations or [],
            previous_programs=previous or [],
        )["user"]

    def test_a_top_program_is_labelled_and_its_neighbour_is_not(self):
        user = self._build(top=[self._prog("bad", 0.9, infeasible=True), self._prog("good", 0.1)])
        header_bad = next(ln for ln in user.splitlines() if "0.9000" in ln)
        header_good = next(ln for ln in user.splitlines() if "0.1000" in ln)
        self.assertIn(LABEL, header_bad)
        self.assertNotIn(LABEL, header_good)

    def test_an_inspiration_is_labelled(self):
        user = self._build(inspirations=[self._prog("bad", 0.9, infeasible=True)])
        self.assertIn(LABEL, user)

    def test_a_previous_attempt_is_labelled_in_its_outcome(self):
        user = self._build(previous=[self._prog("bad", 0.9, infeasible=True)])
        outcome = next(ln for ln in user.splitlines() if ln.startswith("- Outcome:"))
        self.assertIn(LABEL, outcome)

    def test_no_label_anywhere_when_nothing_is_stamped(self):
        """An unconfigured run's prompts must be byte-identical to upstream's."""
        user = self._build(
            top=[self._prog("a", 0.9), self._prog("b", 0.1)],
            inspirations=[self._prog("c", 0.3)],
            previous=[self._prog("d", 0.2)],
        )
        self.assertNotIn(LABEL, user)


if __name__ == "__main__":
    unittest.main()
