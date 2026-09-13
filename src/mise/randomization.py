"""Seeded randomization with pre-policy geometric spawn filtering.

Flat colors and primitive size variation are implemented. Unseen shapes,
textures, reachability and gripper feasibility still need expert validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class RandomizationConfig:
    position_jitter_m: float = 0.05
    yaw_jitter_degrees: float = 30.0
    drawer_open_probability: float = 0.20
    mass_scale: tuple[float, float] = (0.7, 1.3)
    friction_scale: tuple[float, float] = (0.7, 1.3)
    size_scale: tuple[float, float] = (0.90, 1.10)
    light_angle_degrees: float = 40.0
    light_intensity_scale: tuple[float, float] = (0.7, 1.3)
    object_colors: dict[str, tuple[tuple[float, float, float, float], ...]] | None = None
    table_colors: tuple[tuple[float, float, float, float], ...] = ()
    suite: str = "nominal"

    def __post_init__(self) -> None:
        for name in ("position_jitter_m", "yaw_jitter_degrees", "light_angle_degrees"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.drawer_open_probability <= 1:
            raise ValueError("drawer_open_probability must be between zero and one")
        for name in ("mass_scale", "friction_scale", "size_scale", "light_intensity_scale"):
            _pair(getattr(self, name), name)
        if self.suite not in ("nominal", "stress", "held_out"):
            raise ValueError("suite must be nominal, stress, or held_out")

    @classmethod
    def from_yaml(cls, path: Path, *, suite: str | None = None) -> "RandomizationConfig":
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"randomization config must be a mapping: {path}")
        suite = suite or raw.get("suite", "nominal")
        if suite == "stress":
            raw = {**raw, **raw.get("stress", {"mass_scale": [0.5, 1.5], "friction_scale": [0.5, 1.5]})}
        colors = {
            str(name): tuple(tuple(float(channel) for channel in rgba) for rgba in choices)
            for name, choices in raw.get("object_colors", {}).items()
        }
        return cls(
            position_jitter_m=float(raw.get("position_jitter_m", 0.05)),
            yaw_jitter_degrees=float(raw.get("yaw_jitter_degrees", 30.0)),
            drawer_open_probability=float(raw.get("drawer_open_probability", 0.20)),
            mass_scale=_pair(raw.get("mass_scale", (0.7, 1.3)), "mass_scale"),
            friction_scale=_pair(raw.get("friction_scale", (0.7, 1.3)), "friction_scale"),
            size_scale=_pair(raw.get("size_scale", (0.90, 1.10)), "size_scale"),
            light_angle_degrees=float(raw.get("light_angle_degrees", 40.0)),
            light_intensity_scale=_pair(raw.get("light_intensity_scale", (0.7, 1.3)), "light_intensity_scale"),
            object_colors=colors,
            table_colors=tuple(tuple(float(channel) for channel in rgba) for rgba in raw.get("table_colors", [])),
            suite=suite,
        )


def _pair(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two values")
    low, high = (float(item) for item in value)
    if not np.isfinite((low, high)).all() or low <= 0 or high < low:
        raise ValueError(f"{field} must be a finite positive ascending range")
    return low, high


@dataclass(frozen=True, slots=True)
class AppliedRandomization:
    object_positions: dict[str, tuple[float, float, float]]
    drawer_open: bool
    mass_scale: float
    friction_scale: float
    size_scale: float
    light_intensity_scale: float
    suite: str = "nominal"
    rejected_spawn_samples: int = 0
    feasibility_filter: str = "table object separation and drawer bounds; grasp/reach feasibility unvalidated"
    limitations: tuple[str, ...] = (
        "size variation is not unseen-shape generalization",
        "flat colors only; held-out textures and lighting presets are unimplemented",
        "base collision primitive and angular gripper calibration require expert validation",
        "plate and mug are solid cylinders; utensils are thick compound primitives",
    )


class SceneRandomizer:
    """Restore nominal physical properties before every deterministic reset."""

    objects = ("plate", "mug", "bottle", "spoon", "fork")

    def __init__(self, model: Any, config: RandomizationConfig):
        self.model, self.config = model, config
        self._original = {name: getattr(model, name).copy() for name in (
            "body_mass", "body_inertia", "body_ipos", "geom_friction", "geom_size", "geom_pos", "geom_rgba", "light_dir", "light_diffuse",
        )}
        self._object_body_ids = {name: model.body(name).id for name in self.objects}
        self._object_geom_ids = {name: np.flatnonzero(model.geom_bodyid == body) for name, body in self._object_body_ids.items()}
        self._object_joint_addresses = {name: model.joint(f"{name}_free").qposadr[0] for name in self.objects}
        self._drawer_address = model.joint("drawer_joint").qposadr[0]
        self._table_geom_id = model.geom("table_top").id

    def apply(self, data: Any, rng: np.random.Generator) -> AppliedRandomization:
        mass_scale = float(rng.uniform(*self.config.mass_scale))
        friction_scale = float(rng.uniform(*self.config.friction_scale))
        size_scale = float(rng.uniform(*self.config.size_scale))
        light_intensity_scale = float(rng.uniform(*self.config.light_intensity_scale))
        for field, values in self._original.items():
            getattr(self.model, field)[:] = values
        for name in self.objects:
            body_id, geom_ids = self._object_body_ids[name], self._object_geom_ids[name]
            self.model.body_mass[body_id] *= mass_scale
            self.model.body_inertia[body_id] *= mass_scale * size_scale**2
            self.model.body_ipos[body_id] *= size_scale
            self.model.geom_friction[geom_ids] *= friction_scale
            self.model.geom_size[geom_ids] *= size_scale
            self.model.geom_pos[geom_ids] *= size_scale
            choices = (self.config.object_colors or {}).get(name, ())
            if choices:
                self.model.geom_rgba[geom_ids] = choices[int(rng.integers(len(choices)))]
        if self.config.table_colors:
            self.model.geom_rgba[self._table_geom_id] = self.config.table_colors[int(rng.integers(len(self.config.table_colors)))]
        angle = np.deg2rad(rng.uniform(-self.config.light_angle_degrees, self.config.light_angle_degrees))
        self.model.light_dir[0] = np.array((np.sin(angle), 0.25, -np.cos(angle)))
        self.model.light_diffuse[0] *= light_intensity_scale

        drawer_open = bool(rng.random() < self.config.drawer_open_probability)
        drawer_position = 0.18 if drawer_open else 0.0
        data.qpos[self._drawer_address] = drawer_position
        positions: dict[str, tuple[float, float, float]] = {}
        table_nominal = {"plate": (-0.025, -0.23, 0.018, 0.13), "mug": (0.23, -0.23, 0.06, 0.045), "bottle": (-0.30, -0.29, 0.13, 0.032)}
        radii: dict[str, float] = {}
        rejected = 0
        for name, (x, y, height, radius) in table_nominal.items():
            for _ in range(1000):
                xy = np.array((x, y)) + rng.uniform(-self.config.position_jitter_m, self.config.position_jitter_m, size=2)
                valid = abs(xy[0]) + radius * size_scale < 0.61 and abs(xy[1]) + radius * size_scale < 0.51
                valid = valid and all(np.linalg.norm(xy - np.array(p[:2])) > (radius + radii[other]) * size_scale + 0.015 for other, p in positions.items())
                if valid:
                    break
                rejected += 1
            else:
                raise ValueError("spawn feasibility filter exhausted; configuration is invalid before policy execution")
            yaw = np.deg2rad(rng.uniform(-self.config.yaw_jitter_degrees, self.config.yaw_jitter_degrees))
            positions[name] = (float(xy[0]), float(xy[1]), 0.78 + height * size_scale + 0.001)
            radii[name] = radius
            self._place(data, name, positions[name], yaw)
        for name, column in (("spoon", -0.055), ("fork", 0.055)):
            # Partitioned drawer lanes prevent overlap for all accepted yaws.
            # Bounds are checked before physics; no policy outcomes filter seeds.
            for _ in range(1000):
                local = np.array((column, -0.02)) + rng.uniform(-min(self.config.position_jitter_m, 0.008), min(self.config.position_jitter_m, 0.008), size=2)
                yaw = np.deg2rad(rng.uniform(-self.config.yaw_jitter_degrees, self.config.yaw_jitter_degrees))
                corners = np.array([[-0.018, -0.061], [-0.018, 0.101], [0.018, -0.061], [0.018, 0.101]]) * size_scale
                rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                corners = corners @ rot.T + local
                valid = np.max(np.abs(corners[:, 1])) < 0.117 and np.max(np.abs(corners[:, 0])) < 0.132
                valid = valid and (np.max(corners[:, 0]) < -0.003 if column < 0 else np.min(corners[:, 0]) > 0.003)
                if valid:
                    break
                rejected += 1
            else:
                raise ValueError("utensil drawer bounds filter exhausted before policy execution")
            positions[name] = (-0.34 + drawer_position + float(local[0]), 0.27 + float(local[1]), 0.816 + 0.0035 * size_scale + 0.001)
            self._place(data, name, positions[name], yaw)
        return AppliedRandomization(positions, drawer_open, mass_scale, friction_scale, size_scale, light_intensity_scale, self.config.suite, rejected)

    def _place(self, data: Any, name: str, position: tuple[float, float, float], yaw: float) -> None:
        address = self._object_joint_addresses[name]
        data.qpos[address:address + 3] = position
        data.qpos[address + 3:address + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
