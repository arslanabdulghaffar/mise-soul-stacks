"""Deterministic Day-1 expert used as a data-collection and rendering baseline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .sim import BimanualTableEnv
from .types import Observation


class DrawerExpert:
    """Open the drawer while holding the non-active arm in a safe home pose."""

    duration_seconds = 4.0

    def __init__(self, env: BimanualTableEnv):
        self.env = env
        self.home = np.array([0.0, -0.30, 0.72, 0.45, 0.0, 0.35] * 2, dtype=np.float64)
        self.open_pose = np.array([-0.55, -0.65, 1.05, 0.65, 0.0, 0.35], dtype=np.float64)

    def run(self, *, record_hz: int = 5, capture: Callable[[np.ndarray, Observation], None] | None = None) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        steps = round(self.duration_seconds * self.env.control_hz)
        for index in range(steps):
            progress = min(1.0, index / (steps * 0.55))
            action = self.home.copy()
            action[:6] = self.home[:6] * (1.0 - progress) + self.open_pose * progress
            self.env.set_drawer(0.18 * min(1.0, max(0.0, (index - steps * 0.35) / (steps * 0.4))))
            self.env.step(action, observe=False)
            # Physics runs at 30 Hz; sparse capture avoids making a CPU-rendered
            # development video artificially slow on headless machines.
            if index % max(1, self.env.control_hz // record_hz) == 0:
                observation = self.env.observe()
                frames.append(observation.top)
                if capture is not None:
                    capture(action.copy(), observation)
        return frames
