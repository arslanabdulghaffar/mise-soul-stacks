import csv
import json
from pathlib import Path
import tempfile
import unittest

from bench.intel_bench import percentile, write_report


class IntelBenchmarkTests(unittest.TestCase):
    def test_percentile_rejects_empty_measurement(self):
        with self.assertRaises(ValueError):
            percentile([], 95)

    def test_report_keeps_summary_and_raw_timings_separate(self):
        record = {
            "schema_version": "mise.openvino-benchmark.v2",
            "hardware": {"cpu": "test"},
            "component": "contact_act_policy",
            "measured_at_utc": "2026-09-13T00:00:00Z",
            "device": "CPU",
            "device_name": "Test CPU",
            "precision": "f32",
            "model_hash": "a" * 64,
            "measured_iterations": 3,
            "median_ms": 2.0,
            "p95_ms": 2.9,
            "min_ms": 1.0,
            "max_ms": 3.0,
            "throughput_fps": 500.0,
            "compile_ms": 10.0,
            "openvino_version": "test",
            "declared_intel_core_ultra_series": None,
            "intel_core_ultra_series_2_or_3_verified": False,
            "quality_preservation_verified": False,
            "timings_ms": [1.0, 2.0, 3.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(record, root / "latest.json", root / "latest.csv")
            payload = json.loads((root / "latest.json").read_text())
            self.assertEqual(payload["raw_timings_ms"], [1.0, 2.0, 3.0])
            self.assertNotIn("timings_ms", payload["records"][0])
            with (root / "latest.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["p95_ms"], "2.9")


if __name__ == "__main__":
    unittest.main()
