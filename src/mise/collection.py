"""Drawer-actuator integration data, not contact-validated grasp demonstrations."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .datasets import EpisodeSummary, EpisodeWriter, write_dataset_metadata
from .scripted import DrawerExpert
from .sim import BimanualTableEnv

DRAWER_TASK = "open the top drawer"


def collect_drawer_episodes(output_dir: Path, *, episodes: int, seed_start: int = 1000, record_hz: int = 10) -> list[EpisodeSummary]:
    """Collect fixture trajectories; the hidden drawer actuator is not a policy action."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 1 <= record_hz <= BimanualTableEnv.control_hz:
        raise ValueError("record_hz must be between 1 and the control rate")
    write_dataset_metadata(output_dir, task=DRAWER_TASK, control_hz=BimanualTableEnv.control_hz, record_hz=record_hz)
    summaries: list[EpisodeSummary] = []
    env = BimanualTableEnv(seed=seed_start)
    try:
        for episode_index in range(episodes):
            seed = seed_start + episode_index
            env.reset(seed=seed, observe=False)
            writer = EpisodeWriter(output_dir, episode_index=episode_index, task=DRAWER_TASK, seed=seed, control_hz=record_hz)

            def capture(action: np.ndarray, observation: object) -> None:
                # The expert always supplies MISE's Observation at the callback.
                writer.append(observation, action)  # type: ignore[arg-type]

            DrawerExpert(env).run(record_hz=record_hz, capture=capture)
            randomization = asdict(env.last_randomization) if env.last_randomization else {}
            summaries.append(writer.finish(randomization=randomization))
    finally:
        env.close()
    return summaries
