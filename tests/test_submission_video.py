import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_submission_video import select_runs


class SubmissionVideoTests(unittest.TestCase):
    def _run(self, root: Path, seed: int, *, success: bool = True, assisted: bool = False):
        directory = root / str(seed)
        directory.mkdir()
        (directory / "top.mp4").write_bytes(b"video")
        (directory / "run.json").write_text(json.dumps({
            "id": f"run-{seed}", "seed": seed, "preset": "nominal", "status": "completed",
            "scope": "full_task", "created_at": f"2026-09-13T00:00:{seed % 60:02d}Z",
            "summary": {"scope": "full_task", "full_task_success": success, "assisted": assisted},
        }))

    def test_selection_requires_each_predeclared_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1)
            with self.assertRaisesRegex(ValueError, "2"):
                select_runs(root, [1, 2])

    def test_selection_excludes_assisted_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1, assisted=True)
            with self.assertRaises(ValueError):
                select_runs(root, [1])


if __name__ == "__main__":
    unittest.main()
