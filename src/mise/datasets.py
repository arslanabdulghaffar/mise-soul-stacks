"""Portable, LeRobot-aligned raw episode storage.

The project deliberately keeps collection independent of a particular LeRobot
release.  Each compressed episode uses the observation/action key names expected
by LeRobot policies; the later export job can convert these bundles to the
version-pinned Parquet/video layout without losing alignment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Observation

SCHEMA_VERSION = "mise.raw-episode.v1"


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    episode_index: int
    task: str
    frames: int
    path: str
    seed: int


class EpisodeWriter:
    """Collect synchronized three-camera observations, state, and actions."""

    def __init__(self, output_dir: Path, *, episode_index: int, task: str, seed: int, control_hz: int):
        self.output_dir = output_dir
        self.episode_index = episode_index
        self.task = task
        self.seed = seed
        self.control_hz = control_hz
        self._top: list[np.ndarray] = []
        self._wrist_a: list[np.ndarray] = []
        self._wrist_b: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []

    def append(self, observation: Observation, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        state = np.asarray(observation.joint_positions, dtype=np.float32)
        if action.shape != (12,) or state.shape != (12,):
            raise ValueError("episodes require 12-joint state and action vectors")
        self._top.append(np.asarray(observation.top, dtype=np.uint8))
        self._wrist_a.append(np.asarray(observation.wrist_a, dtype=np.uint8))
        self._wrist_b.append(np.asarray(observation.wrist_b, dtype=np.uint8))
        self._state.append(state)
        self._action.append(action)

    def finish(self, *, randomization: dict[str, Any]) -> EpisodeSummary:
        if not self._action:
            raise RuntimeError("cannot write an empty episode")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"episode_{self.episode_index:06d}.npz"
        output = self.output_dir / file_name
        np.savez_compressed(
            output,
            **{
                "observation.images.top": np.stack(self._top),
                "observation.images.wrist_a": np.stack(self._wrist_a),
                "observation.images.wrist_b": np.stack(self._wrist_b),
                "observation.state": np.stack(self._state),
                "action": np.stack(self._action),
                "timestamp": np.arange(len(self._action), dtype=np.float32) / self.control_hz,
            },
        )
        summary = EpisodeSummary(self.episode_index, self.task, len(self._action), file_name, self.seed)
        manifest = self.output_dir / "episodes.jsonl"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**asdict(summary), "randomization": randomization}, sort_keys=True) + "\n")
        return summary


def write_dataset_metadata(output_dir: Path, *, task: str, control_hz: int, record_hz: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "control_hz": control_hz,
        "record_hz": record_hz,
        "observation_keys": [
            "observation.images.top",
            "observation.images.wrist_a",
            "observation.images.wrist_b",
            "observation.state",
        ],
        "action_key": "action",
        "note": "Raw aligned bundles; convert with the version-pinned LeRobot export job before policy training.",
        "controller_provenance": "drawer_actuator_fixture",
        "contact_validated_demonstrations": False,
        "training_limitation": "The drawer is driven by an actuator absent from the 12-joint action vector. These are integration fixtures, not valid robot-grasp demonstrations.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
