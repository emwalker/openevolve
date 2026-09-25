"""Resume seeding: off, a resumed run opens on `random_seed`'s stream exactly as
upstream; on, the stream folds in the loaded iteration, so consecutive resumed
runs draw fresh numbers while a fresh start is unchanged."""

import random
import shutil
import tempfile
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase, resume_seed

SEED = 7


def _config(fold):
    config = Config()
    config.database.in_memory = True
    config.database.random_seed = SEED
    config.database.resume_seed_folds_iteration = fold
    return config.database


def _first_draw(seed):
    return random.Random(seed).random()


class TestResumeSeed(unittest.TestCase):
    def setUp(self):
        self.dirs = {}
        for iteration in (5, 6):
            db = ProgramDatabase(_config(False))
            db.add(Program(id="p", code="x = 1", metrics={"combined_score": 0.5}), iteration=0)
            path = tempfile.mkdtemp()
            db.save(path, iteration=iteration)
            self.dirs[iteration] = path

    def tearDown(self):
        for path in self.dirs.values():
            shutil.rmtree(path, ignore_errors=True)

    def _resumed_draw(self, fold, iteration):
        db = ProgramDatabase(_config(fold))
        db.load(self.dirs[iteration])
        return random.random()

    def test_default_resume_is_upstream(self):
        self.assertEqual(self._resumed_draw(False, 5), _first_draw(SEED))
        self.assertEqual(self._resumed_draw(False, 6), _first_draw(SEED))

    def test_default_construction_from_disk_is_upstream(self):
        config = _config(False)
        config.db_path = self.dirs[5]
        ProgramDatabase(config)
        self.assertEqual(random.random(), _first_draw(SEED))

    def test_fresh_start_is_unchanged_when_on(self):
        ProgramDatabase(_config(True))
        self.assertEqual(random.random(), _first_draw(SEED))

    def test_resume_folds_the_iteration_when_on(self):
        at_five = self._resumed_draw(True, 5)
        self.assertEqual(at_five, _first_draw(resume_seed(SEED, 5)))
        self.assertNotEqual(at_five, _first_draw(SEED))
        self.assertNotEqual(at_five, self._resumed_draw(True, 6))

    def test_construction_from_disk_folds_when_on(self):
        config = _config(True)
        config.db_path = self.dirs[6]
        ProgramDatabase(config)
        self.assertEqual(random.random(), _first_draw(resume_seed(SEED, 6)))


if __name__ == "__main__":
    unittest.main()
