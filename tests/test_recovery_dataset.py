from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mise.recovery_dataset import collect_drawer_fork_labels


class RecoveryDatasetTests(unittest.TestCase):
    def test_bootstrap_job_emits_all_three_fork_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = collect_drawer_fork_labels(Path(temporary), samples=3)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["decision"]["verdict"] for row in rows], ["continue", "replan", "stop"])
        self.assertTrue(all(row["fork_statistics"]["forks_per_arm"] == 24 for row in rows))


if __name__ == "__main__":
    unittest.main()
