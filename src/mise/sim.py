"""MuJoCo wrapper exposing camera pixels and joint angles, not privileged state."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from .randomization import AppliedRandomization, RandomizationConfig, SceneRandomizer
from .types import Observation

ROOT: Final = Path(__file__).resolve().parents[2]
SCENE_PATH: Final = ROOT / "assets" / "generated" / "mise_bimanual.xml"
RANDOMIZATION_PATH: Final = ROOT / "configs" / "scene_randomization.yaml"
ARM_JOINTS: Final = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class MissingMuJoCoError(RuntimeError):
    pass


def build_scene() -> Path:
    """Generate the namespaced two-arm scene if it does not exist."""

    from runpy import run_path
    return run_path(str(ROOT / 'scripts/build_scene.py'))['build_scene']()


class BimanualTableEnv:
    control_hz = 30
    physics_hz = 500
    render_size = 256

    def __init__(self, *, seed: int = 0, scene_path: Path | None = None, randomization_path: Path | None = None,
                 randomize: bool = True, initial_observation: bool = True):
        try:
            import mujoco
        except ImportError as exc:
            raise MissingMuJoCoError("MuJoCo is required; install the project with `python -m pip install -e .`.") from exc
        path = scene_path or (SCENE_PATH if SCENE_PATH.exists() else build_scene())
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._randomize = randomize
        self._render_options = mujoco.MjvOption()
        # The official model has coincident visual and collision meshes. Hide
        # collision group 3 for cameras only; physics contacts remain enabled.
        self._render_options.geomgroup[3] = 0
        self._rng = np.random.default_rng(seed)
        config_path = randomization_path or RANDOMIZATION_PATH
        config = RandomizationConfig.from_yaml(config_path) if config_path.exists() else RandomizationConfig()
        self.randomizer = SceneRandomizer(self.model, config)
        self.last_randomization: AppliedRandomization | None = None
        self._arm_qpos = np.array(
            [
                [self.model.joint(f"arm_{arm}_{joint}").qposadr[0] for joint in ARM_JOINTS]
                for arm in ("a", "b")
            ],
            dtype=int,
        )
        self._arm_ctrl = np.array(
            [
                [self.model.actuator(f"arm_{arm}_{joint}").id for joint in ARM_JOINTS]
                for arm in ("a", "b")
            ],
            dtype=int,
        )
        self.reset(seed=seed, observe=initial_observation)

    def reset(self, *, seed: int | None = None, observe: bool = True) -> Observation | None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        self._physics_credit = 0
        self.last_randomization = self.randomizer.apply(self.data, self._rng) if self._randomize else None
        self.data.ctrl[:] = self.model.actuator_ctrlrange[:, 0] * 0.0
        self._mj.mj_forward(self.model, self.data)
        return self.observe() if observe else None

    def step(self, action: np.ndarray, *, frames: int | None = None, observe: bool = True) -> Observation | None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,):
            raise ValueError("action must be 12 joint targets ordered arm A then arm B")
        if not np.isfinite(action).all():
            raise ValueError("joint targets must be finite")
        if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 1):
            raise ValueError("frames must be a positive integer")
        for arm_index in range(2):
            ctrl_ids = self._arm_ctrl[arm_index]
            low = self.model.actuator_ctrlrange[ctrl_ids, 0]
            high = self.model.actuator_ctrlrange[ctrl_ids, 1]
            self.data.ctrl[ctrl_ids] = np.clip(action[arm_index * 6 : (arm_index + 1) * 6], low, high)
        # Policies act at 30 Hz while MuJoCo integrates at 500 Hz, matching the
        # project brief.  Tests may supply an explicit substep count when needed.
        if frames is None:
            self._physics_credit += self.physics_hz
            substeps, self._physics_credit = divmod(self._physics_credit, self.control_hz)
        else:
            substeps = frames
        for _ in range(substeps):
            self._mj.mj_step(self.model, self.data)
        return self.observe() if observe else None

    def set_drawer(self, position: float) -> None:
        actuator = self.model.actuator("drawer_slide")
        low, high = self.model.actuator_ctrlrange[actuator.id]
        self.data.ctrl[actuator.id] = np.clip(position, low, high)

    def observe(self) -> Observation:
        return Observation(
            top=self.render("top"),
            wrist_a=self.render("wrist_a"),
            wrist_b=self.render("wrist_b"),
            joint_positions=self.joint_positions(),
        )

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._arm_qpos.ravel()].copy()

    def render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, height=self.render_size, width=self.render_size)
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._render_options)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
