import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from scripts.build_submission_video import main, select_runs, sha256, verify_run_files


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
        for name in ('summary.json', 'trace.jsonl', 'timing.csv', 'wrist_a.mp4', 'wrist_b.mp4'):
            (directory / name).write_text('fixture')
        (directory / 'manifest.json').write_text(json.dumps({
            'run_id': f'run-{seed}', 'scope': 'full_task',
            'files_sha256': {p.name: sha256(p) for p in directory.iterdir()},
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

    def test_check_only_preserves_existing_video_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(10):
                self._run(root, seed)
            seeds = root / 'seeds.yaml'
            seeds.write_text('required: [0,1,2,3,4,5,6,7,8,9]\n')
            manifest = root / 'video-manifest.json'
            original = b'{"video": "original.mp4", "video_sha256": "original-binding"}\n'
            manifest.write_bytes(original)
            args = ['video', '--runs-root', str(root), '--seeds', str(seeds),
                    '--manifest', str(manifest), '--check-only']
            with patch('sys.argv', args), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(json.loads(output.getvalue())['verified_artifact_files'], 70)

    def test_tampered_evidence_cannot_qualify_for_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, 1)
            selected, _ = select_runs(root, [1])
            self.assertEqual(verify_run_files(selected[0]), 7)
            (root / '1' / 'top.mp4').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_run_files(selected[0])


if __name__ == "__main__":
    unittest.main()
