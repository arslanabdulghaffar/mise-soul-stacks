"""Installation gates must reject changed weights, scenes, and runtime code."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mise.learned_assets import accepted_policy
from mise.telemetry import file_hash


class LearnedAssetTests(unittest.TestCase):
    def test_registration_binds_every_model_file_and_runtime_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            (model / 'openvino').mkdir(parents=True)
            files = {}
            for name in ('contact_act.xml', 'contact_act.bin', 'model_manifest.json'):
                path = model / 'openvino' / name
                path.write_text('tested contents')
                files[name] = file_hash(path)
            (root / 'scene.xml').write_text('scene')
            (root / 'controller.py').write_text('runtime')
            record = dict(schema_version='mise.accepted-contact-policy.v1', successes=10, total=10,
                          model_files_sha256=files, scene_path='scene.xml',
                          scene_sha256=file_hash(root / 'scene.xml'),
                          runtime_sources_sha256={'controller.py': file_hash(root / 'controller.py')})
            registration = model / 'accepted_policy.json'
            registration.write_text(json.dumps(record))
            with patch('mise.learned_assets.ROOT', root):
                self.assertEqual(accepted_policy(model)['successes'], 10)
                for path in [*(model / 'openvino' / name for name in files), root / 'scene.xml', root / 'controller.py']:
                    original = path.read_bytes()
                    path.write_text('changed')
                    with self.assertRaises(ValueError):
                        accepted_policy(model)
                    path.write_bytes(original)
                record['model_files_sha256'] = {}
                registration.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    accepted_policy(model)
