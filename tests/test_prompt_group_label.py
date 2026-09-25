"""An inspiration from another group is labelled with that group, or its score
reads as one earned under the parent's own group."""

import unittest

from openevolve.config import Config
from openevolve.database import Program
from openevolve.process_parallel import _label_other_groups
from openevolve.prompt.sampler import PromptSampler

NOTE = "From group"


def _prog(pid, lane):
    return Program(id=pid, code=f"def {pid}(): pass", metrics={"combined_score": 0.4, "lane": lane})


class TestGroupLabel(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.prompt.num_top_programs = 0
        config.prompt.num_diverse_programs = 2
        config.prompt.include_artifacts = False
        self.sampler = PromptSampler(config.prompt)

    def _build(self, inspirations):
        return self.sampler.build_prompt(
            current_program="def cur(): pass",
            parent_program="def cur(): pass",
            program_metrics={"combined_score": 0.5},
            inspirations=inspirations,
        )["user"]

    def test_only_other_groups_are_stamped_with_their_name(self):
        programs = [_prog("own", 1.0), _prog("other", 2.0), _prog("unnamed", 3.0)]
        payloads = [p.to_dict() for p in programs]
        _label_other_groups(payloads, programs, 1, "lane", {2: "second"})
        self.assertNotIn("other_group", payloads[0])
        self.assertEqual(payloads[1]["other_group"], "second")
        self.assertEqual(payloads[2]["other_group"], "3")

    def test_a_stamped_inspiration_is_labelled_under_its_header(self):
        own, other = _prog("own", 1.0).to_dict(), _prog("other", 2.0).to_dict()
        other["other_group"] = "second"
        lines = self._build([own, other]).splitlines()
        header = next(i for i, ln in enumerate(lines) if ln.startswith("### Inspiration 2"))
        self.assertTrue(lines[header + 1].startswith(f"{NOTE} second:"))
        self.assertEqual(sum(NOTE in ln for ln in lines), 1)

    def test_unstamped_prompt_is_upstream(self):
        """Nothing stamped renders as the upstream template does, byte for byte."""
        inspirations = [_prog("a", 1.0).to_dict(), _prog("b", 2.0).to_dict()]
        ours = self._build(inspirations)
        manager = self.sampler.template_manager
        template = manager.get_template("inspiration_program")
        manager.add_template("inspiration_program", template.replace("{group_note}", ""))
        self.assertEqual(ours, self._build(inspirations))
        self.assertNotIn(NOTE, ours)


if __name__ == "__main__":
    unittest.main()
