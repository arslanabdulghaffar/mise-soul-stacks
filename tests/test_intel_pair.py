import json
from pathlib import Path
import tempfile
import unittest

from mise.learning.openvino_config import compile_config, detected_core_ultra_series
from mise.telemetry import file_hash
from scripts.validate_intel_pair import compare_pair, read_evaluation


class IntelPairTests(unittest.TestCase):
    def test_series_requires_actual_sku_not_declaration_or_brand_only(self):
        for cpu, expected in [('Intel(R) Core(TM) Ultra 7 155H', 1),
                              ('Intel(R) Core(TM) Ultra 7 258V', 2),
                              ('Intel Core Ultra X7 Processor 358H', 3),
                              ('Intel Core Ultra 7 Processor 355', 3),
                              ('Intel Xeon Platinum 8581C', None),
                              ('Intel Core Ultra', None)]:
            self.assertEqual(detected_core_ultra_series(cpu), expected)

    def test_gpu_does_not_receive_cpu_only_thread_property(self):
        self.assertEqual(compile_config('CPU', 'f32', 3)['INFERENCE_NUM_THREADS'], 3)
        self.assertNotIn('INFERENCE_NUM_THREADS', compile_config('GPU.0', 'f16', 3))
        self.assertNotIn('INFERENCE_PRECISION_HINT', compile_config('CPU', 'auto'))
        with self.assertRaises(ValueError):
            compile_config('CPU', 'f32', 0)

    def _pair(self, root, *, successes=True, count=10, cpu='Intel Core Ultra 7 258V'):
        paths = []
        hardware = dict(cpu=cpu, hostname='target', versions={'openvino': 'test'})
        for name, precision, latency in [('reference', 'f32', 4.), ('candidate', 'f16', 2.)]:
            directory = root / name
            directory.mkdir()
            hashes = dict.fromkeys(['contact_act.xml', 'contact_act.bin', 'model_manifest.json'], 'a' * 64)
            manifest = dict(schema_version='mise.learned-contact-eval.v2', seeds=list(range(count)),
                            backend='openvino', files_sha256=hashes, command='mug', max_steps=1200,
                            scope='learned_mug_contact_skill', hardware=hardware,
                            runtime_config=dict(device='CPU', precision=precision,
                                compile_config=compile_config('CPU', precision), resolved_properties={'test': 'ok'}))
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            attempts, attempt_hashes = [], {}
            for seed in manifest['seeds']:
                result = dict(seed=seed, success=successes, controller_done=successes,
                              physics={'success': successes})
                path = directory / f'{seed}.json'
                path.write_text(json.dumps(dict(result=result, trace=[{'time_s': 1.}], inference_timings_ms=[1.])))
                attempt_hashes[path.name] = file_hash(path)
                attempts.append(result)
            results = dict(attempts=attempts, complete=True, total=count, predeclared=count,
                           successes=count if successes else 0, files_sha256=attempt_hashes,
                           manifest_sha256=file_hash(directory / 'manifest.json'))
            (directory / 'results.json').write_text(json.dumps(results))
            record = dict(hardware=hardware, model_hash='a' * 64, weights_hash='a' * 64,
                          model_manifest_hash='a' * 64, component='contact_act_policy', openvino_version='test-build',
                          measured_iterations=100, warmup_iterations=10, input_seed=1701,
                          median_ms=latency, p95_ms=latency, declared_intel_core_ultra_series=2,
                          intel_core_ultra_series_2_or_3_verified=True, **manifest['runtime_config'])
            benchmark = root / f'{name}.json'
            benchmark.write_text(json.dumps(dict(records=[record], raw_timings_ms=[latency] * 100)))
            paths.extend([directory, benchmark])
        return paths[0], paths[2], paths[1], paths[3]

    def test_complete_matching_pair_can_pass_strict_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            report = compare_pair(*self._pair(Path(temp)))
            self.assertTrue(report['submission_optimization_verified'])
            self.assertEqual(report['measured_median_speedup'], 2.)
            self.assertIsNone(report['full_task_success'])

    def test_zero_success_or_small_pilot_never_verifies_quality(self):
        for kwargs in ({'successes': False}, {'count': 1}):
            with tempfile.TemporaryDirectory() as temp:
                report = compare_pair(*self._pair(Path(temp), **kwargs))
                self.assertFalse(report['quality_preservation_verified'])
                self.assertFalse(report['submission_optimization_verified'])

    def test_declared_target_cannot_promote_a_xeon_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            report = compare_pair(*self._pair(Path(temp), cpu='Intel Xeon Platinum 8480+'))
            self.assertTrue(report['quality_preservation_verified'])
            self.assertFalse(report['intel_core_ultra_series_2_or_3_verified'])
            self.assertFalse(report['submission_optimization_verified'])

    def test_modified_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ref, *_ = self._pair(Path(temp))
            path = ref / '0.json'
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                read_evaluation(ref)

    def test_mismatched_precision_or_benchmark_summary_is_rejected(self):
        for field, value in [('precision', 'bf16'), ('median_ms', 999.)]:
            with tempfile.TemporaryDirectory() as temp:
                paths = self._pair(Path(temp))
                path = paths[3]
                payload = json.loads(path.read_text())
                payload['records'][0][field] = value
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    compare_pair(*paths)

    def test_incomplete_or_different_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self._pair(Path(temp))
            path = paths[1] / 'results.json'
            results = json.loads(path.read_text())
            results['complete'] = False
            path.write_text(json.dumps(results))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                compare_pair(*paths)

    def test_controller_configuration_change_is_not_an_inference_optimization(self):
        for field, value in [('execution_steps', 5), ('temporal_ensemble', True)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                paths = self._pair(Path(temp))
                manifest_path = paths[1] / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest[field] = value
                manifest_path.write_text(json.dumps(manifest))
                results_path = paths[1] / 'results.json'
                results = json.loads(results_path.read_text())
                results['manifest_sha256'] = file_hash(manifest_path)
                results_path.write_text(json.dumps(results))
                with self.assertRaisesRegex(ValueError, f'controller settings differ: {field}'):
                    compare_pair(*paths)


if __name__ == '__main__':
    unittest.main()
