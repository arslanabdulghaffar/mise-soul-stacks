"""Privileged, read-only task evaluator; never pass its state to a policy.

Geometric task predicates are useful before manipulation is solved. Certification
also needs actual contact provenance and validated collision/gripper geometry.
A drawer fixture opening alone therefore cannot certify an arm-A manipulation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "evaluator.yaml"


class StablePredicate:
    """Require a continuous observed dwell, with gaps and resets breaking it.

    This public helper is shared by offline checks and external consumers. A
    timestamp gap cannot establish that the predicate held while unobserved.
    """

    def __init__(self, duration: float, max_gap: float = 0.1):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        if not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be finite and positive")
        self.duration, self.max_gap = float(duration), float(max_gap)
        self.since: float | None = None
        self.last: float | None = None

    def update(self, time: float, satisfied: bool) -> bool:
        if not np.isfinite(time):
            raise ValueError("non-finite observation time cannot establish stability")
        if self.last is not None and (time < self.last or time - self.last > self.max_gap + 1e-9):
            self.since = None
        self.last = float(time)
        if not satisfied:
            self.since = None
            return False
        if self.since is None:
            self.since = float(time)
        return time - self.since >= self.duration - 1e-9


def wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    """Two-sided 95% Wilson score interval for recorded Bernoulli outcomes."""

    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (successes, total)):
        raise ValueError("successes and total must be integer counts")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


@dataclass(frozen=True, slots=True)
class GoalRegion:
    name: str
    center_xy: tuple[float, float]
    radius_m: float
    height_range_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    drawer_open_fraction: float = 0.8
    drawer_hold_s: float = 0.5
    placement_hold_s: float = 1.0
    stability_speed_m_s: float = 0.02
    stability_angular_speed_rad_s: float = 0.10
    mug_max_tilt_degrees: float = 15.0
    utensil_max_axis_error_degrees: float = 20.0
    handoff_hold_s: float = 1.0
    episode_timeout_s: float = 180.0
    object_loss_height_m: float = 0.50
    max_sample_gap_s: float = 0.10
    gripper_contacts_validated: bool = False
    collision_geometry_validated: bool = False
    goal_regions: dict[str, GoalRegion] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> "EvaluatorConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("evaluator configuration must be a mapping")
        regions = {name: GoalRegion(value["name"], tuple(value["center_xy"]), float(value["radius_m"]), tuple(value["height_range_m"])) for name, value in raw.pop("goal_regions").items()}
        config = cls(**raw, goal_regions=regions)
        if not 0 < config.drawer_open_fraction <= 1:
            raise ValueError("drawer_open_fraction must be in (0, 1]")
        for name in ("drawer_hold_s", "placement_hold_s", "stability_speed_m_s", "stability_angular_speed_rad_s", "handoff_hold_s", "episode_timeout_s", "max_sample_gap_s"):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= config.mug_max_tilt_degrees <= 180:
            raise ValueError("mug tilt tolerance must be in [0, 180]")
        if not 0 <= config.utensil_max_axis_error_degrees <= 90:
            raise ValueError("utensil axis tolerance must be in [0, 90]")
        if set(regions) != {"plate", "fork", "spoon", "mug"}:
            raise ValueError("goal_regions must declare plate, fork, spoon and mug")
        for region in regions.values():
            if len(region.center_xy) != 2 or len(region.height_range_m) != 2 or not np.isfinite((*region.center_xy, *region.height_range_m, region.radius_m)).all() or region.radius_m <= 0 or region.height_range_m[0] >= region.height_range_m[1]:
                raise ValueError("invalid goal region dimensions")
        return config


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    elapsed_s: float
    drawer_open: bool
    objects_in_goal: dict[str, bool]
    handoff_complete: bool
    full_task_success: bool
    forbidden_collision_count: int
    violations: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskEvaluator:
    """Episode-scoped observer. Construct a new evaluator after an env reset.

    Call at <=100 ms intervals (30 Hz normally). Missing samples interrupt dwell
    windows, and a backwards simulator clock is a reset violation. The evaluator
    reads contacts and poses only; it does not alter model, data or controls.
    """

    def __init__(self, model: Any, config_path: Path | None = None, *, config: EvaluatorConfig | None = None):
        import mujoco
        self._mj, self.model = mujoco, model
        self.config = config or EvaluatorConfig.from_yaml(config_path or CONFIG_PATH)
        self._bodies = {name: model.body(name).id for name in self.config.goal_regions}
        self._drawer_address = model.joint("drawer_joint").qposadr[0]
        self._drawer_range = model.joint("drawer_joint").range.copy()
        self._started: float | None = None
        self._last_time: float | None = None
        self._since: dict[str, float] = {}
        self._violations: set[str] = set()
        self._active_collisions: set[tuple[int, int]] = set()
        self._collision_count = 0
        self._handoff_phase = "awaiting_A"
        self._handoff_complete = False
        self._drawer_arm_a = False
        self._utensils_started_in_drawer: set[str] = set()
        self._retrieved: set[str] = set()

    def _dwell(self, key: str, condition: bool, now: float, duration: float) -> bool:
        if not condition:
            self._since.pop(key, None)
            return False
        self._since.setdefault(key, now)
        return now - self._since[key] >= duration - 1e-9

    def update(self, data: Any, *, violations: Iterable[str] = ()) -> EvaluationSnapshot:
        now = float(data.time)
        if not np.isfinite(now):
            raise ValueError("non-finite simulator time cannot be evaluated")
        if self._started is None:
            self._started = now
            drawer_position = np.asarray(data.body("drawer").xpos)
            for name in ("spoon", "fork"):
                offset = np.asarray(data.body(name).xpos) - drawer_position
                if abs(offset[0]) < 0.14 and abs(offset[1]) < 0.125 and 0.005 <= offset[2] <= 0.09:
                    self._utensils_started_in_drawer.add(name)
        if self._last_time is not None:
            if now < self._last_time:
                self._violations.add("simulator_reset")
                self._since.clear()
                self._handoff_phase = "awaiting_A"
                self._handoff_complete = False
            elif now - self._last_time > self.config.max_sample_gap_s + 1e-9:
                self._since.clear()
                self._handoff_phase = "awaiting_A"
        self._last_time = now
        self._violations.update(violations)
        elapsed = max(0.0, now - self._started)
        if elapsed > self.config.episode_timeout_s:
            self._violations.add("timeout")
        contacts = self._contacts(data)
        opening = (float(data.qpos[self._drawer_address]) - self._drawer_range[0]) / np.ptp(self._drawer_range)
        self._drawer_arm_a |= contacts["drawer_A"] and opening > 0.05
        drawer_open = self._dwell("drawer", opening >= self.config.drawer_open_fraction, now, self.config.drawer_hold_s)
        goals, object_evidence = {}, {}
        for name, region in self.config.goal_regions.items():
            body_id = self._bodies[name]
            position = data.xpos[body_id]
            velocity = np.zeros(6)
            self._mj.mj_objectVelocity(self.model, data, self._mj.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            linear_speed, angular_speed = float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))
            tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[body_id].reshape(3, 3)[2, 2], -1, 1))))
            local_length_axis = data.xmat[body_id].reshape(3, 3)[:, 1]
            axis_error = float(np.rad2deg(np.arctan2(abs(local_length_axis[0]), abs(local_length_axis[1]))))
            error = float(np.linalg.norm(position[:2] - region.center_xy))
            inside = error <= region.radius_m and region.height_range_m[0] <= position[2] <= region.height_range_m[1]
            released = not contacts["arm_touch"][name]
            stable = linear_speed <= self.config.stability_speed_m_s and angular_speed <= self.config.stability_angular_speed_rad_s
            upright = name != "mug" or tilt <= self.config.mug_max_tilt_degrees
            aligned = name not in {"fork", "spoon"} or axis_error <= self.config.utensil_max_axis_error_degrees
            goals[name] = self._dwell(name, inside and released and stable and upright and aligned and contacts["table_support"][name], now, self.config.placement_hold_s)
            if position[2] < self.config.object_loss_height_m:
                self._violations.add(f"object_lost:{name}")
            if name in self._utensils_started_in_drawer and contacts["held"][name] and not contacts["supported"][name]:
                self._retrieved.add(name)
            object_evidence[name] = {"region": region.name, "center_error_m": error, "height_m": float(position[2]), "linear_speed_m_s": linear_speed, "angular_speed_rad_s": angular_speed, "tilt_degrees": tilt, "axis_error_degrees": axis_error if name in {"fork", "spoon"} else None, "released": released, "table_supported": contacts["table_support"][name], "held_by": sorted(contacts["held"][name]), "stable_duration_s": max(0, now - self._since.get(name, now))}
        self._update_handoff(contacts["held"]["spoon"], contacts["supported"]["spoon"], now)
        self._collision_count += len(contacts["forbidden"] - self._active_collisions)
        self._active_collisions = contacts["forbidden"]
        if self._collision_count:
            self._violations.add("forbidden_collision")
        certified = self.config.gripper_contacts_validated and self.config.collision_geometry_validated
        full = drawer_open and all(goals.values()) and self._handoff_complete and self._drawer_arm_a and {"spoon", "fork"} <= self._retrieved and certified and not self._violations
        return EvaluationSnapshot(elapsed, drawer_open, goals, self._handoff_complete, bool(full), self._collision_count, tuple(sorted(self._violations)), {
            "role": "privileged_evaluator", "geometry_validated": bool(certified),
            "drawer_open_fraction": float(opening), "drawer_arm_A_contact_observed": bool(self._drawer_arm_a),
            "objects": object_evidence, "utensils_retrieved_from_drawer": sorted(self._retrieved),
            "handoff_phase": self._handoff_phase, "gripper_contacts_validated": self.config.gripper_contacts_validated,
            "collision_geometry_validated": self.config.collision_geometry_validated,
            "certification_limit": "contact geometry and expert grasp validation required" if not certified else None,
        })

    def _update_handoff(self, held: set[str], supported: bool, now: float) -> None:
        if self._handoff_complete:
            return
        if supported:
            self._handoff_phase = "awaiting_A"
            self._since.pop("handoff", None)
        elif self._handoff_phase == "awaiting_A" and held == {"A"}:
            self._handoff_phase = "A_holds"
        elif self._handoff_phase == "A_holds":
            if held == {"A", "B"}:
                self._handoff_phase = "both_hold"
            elif held != {"A"}:
                self._handoff_phase = "awaiting_A"
        elif self._handoff_phase in ("both_hold", "B_retains"):
            if held == {"B"}:
                self._handoff_phase = "B_retains"
                self._handoff_complete = self._dwell("handoff", True, now, self.config.handoff_hold_s)
            elif held != {"A", "B"}:
                self._handoff_phase = "awaiting_A"
                self._since.pop("handoff", None)
            else:
                self._since.pop("handoff", None)

    def _contacts(self, data: Any) -> dict[str, Any]:
        names = tuple(self._bodies)
        result: dict[str, Any] = {"arm_touch": {name: set() for name in names}, "held": {name: set() for name in names}, "supported": {name: False for name in names}, "table_support": {name: False for name in names}, "drawer_A": False, "forbidden": set()}
        jaw_contacts = {name: {arm: set() for arm in ("A", "B")} for name in names}
        body_to_object = {body: name for name, body in self._bodies.items()}
        for index in range(data.ncon):
            contact = data.contact[index]
            force = np.zeros(6)
            self._mj.mj_contactForce(self.model, data, index, force)
            if force[0] <= 0.001:
                continue
            geom_ids = (int(contact.geom1), int(contact.geom2))
            body_ids = tuple(int(self.model.geom_bodyid[geom]) for geom in geom_ids)
            body_names = tuple(self.model.body(body).name for body in body_ids)
            geom_names = tuple(self.model.geom(geom).name for geom in geom_ids)
            arms = tuple("A" if name.startswith("arm_a_") else "B" if name.startswith("arm_b_") else None for name in body_names)
            if all(arms) and arms[0] != arms[1]:
                result["forbidden"].add(tuple(sorted(geom_ids)))
            for side in (0, 1):
                other = 1 - side
                name = body_to_object.get(body_ids[side])
                arm = arms[other]
                if name:
                    if arm:
                        result["arm_touch"][name].add(arm)
                        prefix = f"arm_{arm.lower()}_"
                        if body_names[other] == prefix + "gripper":
                            jaw_contacts[name][arm].add("fixed")
                        elif body_names[other] == prefix + "moving_jaw_so101_v1":
                            jaw_contacts[name][arm].add("moving")
                        else:
                            result["forbidden"].add(tuple(sorted(geom_ids)))
                    else:
                        # Conservatively treat any non-arm contact as support;
                        # a brush against the drawer cannot qualify as handoff.
                        result["supported"][name] = True
                        result["table_support"][name] |= geom_names[other] == "table_top"
                if arms[side] and not arms[other] and name is None:
                    target_object = body_to_object.get(body_ids[other])
                    if geom_names[other] == "drawer_handle" and body_names[side] in (f"arm_{arms[side].lower()}_gripper", f"arm_{arms[side].lower()}_moving_jaw_so101_v1"):
                        result["drawer_A"] |= arms[side] == "A"
                    elif target_object is None:
                        result["forbidden"].add(tuple(sorted(geom_ids)))
        for name in names:
            for arm in ("A", "B"):
                if jaw_contacts[name][arm] == {"fixed", "moving"}:
                    result["held"][name].add(arm)
        return result
