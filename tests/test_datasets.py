from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mise.datasets import EpisodeWriter, write_dataset_metadata
from mise.types import Observation


class DatasetTests(unittest.TestCase):
    def test_episode_writer_keeps_all_modalities_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            writer = EpisodeWriter(output, episode_index=0, task="open drawer", seed=1, control_hz=10)
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            observation = Observation(image, image + 1, image + 2, np.arange(12, dtype=np.float32))
            writer.append(observation, np.ones(12, dtype=np.float32))
            writer.append(observation, np.full(12, 2, dtype=np.float32))
            summary = writer.finish(randomization={"drawer_open": False})
            payload = np.load(output / summary.path)
            self.assertEqual(payload["observation.images.top"].shape, (2, 4, 4, 3))
            self.assertEqual(payload["observation.state"].shape, (2, 12))
            self.assertEqual(payload["action"].shape, (2, 12))
            write_dataset_metadata(output, task="open drawer", control_hz=30, record_hz=10)
            self.assertEqual(json.loads((output / "metadata.json").read_text())["schema_version"], "mise.raw-episode.v1")
