"""Camera-grounded execution of the complete bimanual table-setting plan."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from runpy import run_path

import numpy as np

from .evaluation import TaskEvaluator
from .full_vision import locate_full_mug, locate_handle, locate_plate, locate_utensil
from .kinematics import ArmKinematics
from .manipulation import HOME
from .planner import DEFAULT_COMMAND, validate_plan_graph
from .recovery_memory import AdaptiveRecoverySupervisor, RecoveryContext, RecoveryMemory
from .sim import BimanualTableEnv, ROOT
from .supervisor import MonitorEvidence, RecoveryCandidate, RecoverySupervisor
from .types import RecoverabilityVerdict

FULL_COMMANDS = ("Set the table.", DEFAULT_COMMAND)
FULL_CONFIG = ROOT / "configs/full_evaluator.yaml"
RECOVERY_MODES = ("none", "blind_retry", "adaptive")
_SIGNATURES = (
    ("open_drawer", "A", "drawer", "drawer_open", None, None),
    ("pick_place", "A", "plate", "table_center", None, None),
    ("pick", "A", "spoon", "held_by_A", None, None),
    ("handoff", "both", "spoon", "held_by_B", "A", "B"),
    ("pick_place", "B", "spoon", "table_right", None, None),
    ("pick_place", "A", "fork", "table_left", None, None),
    ("pick_place", "B", "mug", "table_upper_right", None, None),
)


@dataclass(frozen=True, slots=True)
class FullSceneRandomization:
    """Small pre-policy variations within the validated expert envelope."""

    object_offsets_m: dict[str, tuple[float, float]]
    mass_scale: float
    friction_scale: float
    suite: str = "full_task_nominal"
    feasibility_filter: str = "fixed per-object jitter bounds inside the validated shared scene"
    limitations: tuple[str, ...] = (
        "primitive shapes and fixed visual fiducial colors",
        "small position and physical-property variation only",
        "not evidence of unseen-shape or real-hardware generalization",
    )


def is_full_plan(plan) -> bool:
    return tuple((step.skill, step.arm, step.object, step.target, step.donor, step.receiver)
                 for step in plan.steps) == _SIGNATURES


def validate_full_plan(plan) -> None:
    validate_plan_graph(plan)
    if not is_full_plan(plan):
        raise ValueError("The complete-task controller requires the registered seven-step table-setting plan.")


def create_full_env(seed: int = 0) -> BimanualTableEnv:
    scene = run_path(str(ROOT / "scripts/build_full_scene.py"))["build_full_scene"]()
    env = BimanualTableEnv(seed=seed, scene_path=scene, randomize=False, initial_observation=False)
    env.data.qpos[env._arm_qpos.ravel()] = HOME
    env.data.ctrl[env._arm_ctrl.ravel()] = HOME
    rng = np.random.default_rng(seed)
    limits = {"plate": .006, "mug": .006, "spoon": .003, "fork": .003}
    offsets = {name: tuple(float(value) for value in rng.uniform(-limit, limit, size=2))
               for name, limit in limits.items()}
    mass_scale = float(rng.uniform(.97, 1.03))
    friction_scale = float(rng.uniform(.95, 1.05))
    for name, offset in offsets.items():
        address = env.model.joint(f"{name}_free").qposadr[0]
        env.data.qpos[address:address + 2] += offset
        body = env.model.body(name).id
        env.model.body_mass[body] *= mass_scale
        env.model.body_inertia[body] *= mass_scale
        env.model.geom_friction[env.model.geom_bodyid == body] *= friction_scale
    env.last_randomization = FullSceneRandomization(offsets, mass_scale, friction_scale)
    env._mj.mj_forward(env.model, env.data)
    for _ in range(300):
        env._mj.mj_step(env.model, env.data)
    return env


class _MotionProgram:
    """Compose smooth joint targets using robot FK/IK and controller history."""

    def __init__(self, env: BimanualTableEnv):
        self.env = env
        self.action = env.joint_positions()
        self.ik = {arm: ArmKinematics(env.model, arm) for arm in "AB"}
        for arm in "AB":
            self.ik[arm].site = env.model.site(f"arm_{arm.lower()}_narrow_tcp").id
        self.q = {arm: self.action[slice(0, 6) if arm == "A" else slice(6, 12)].copy() for arm in "AB"}
        self.last_point: dict[str, np.ndarray | None] = {"A": None, "B": None}
        self.last_yaw: dict[str, float | None] = {"A": None, "B": None}

    def use_tcp(self, arm: str, narrow: bool) -> None:
        suffix = "narrow_tcp" if narrow else "tcp"
        self.ik[arm].site = self.env.model.site(f"arm_{arm.lower()}_{suffix}").id
        self.last_point[arm] = None
        self.last_yaw[arm] = None

    def seed(self, arm: str, values) -> None:
        self.q[arm] = np.asarray(values, dtype=float).copy()
        self.last_point[arm] = None
        self.last_yaw[arm] = None

    def move(self, arm: str, phase: str, point=None, *, grip=None, seconds=2.0,
             yaw: float | None = 0.0) -> tuple[str, np.ndarray]:
        arm_slice = slice(0, 6) if arm == "A" else slice(6, 12)
        q, before = self.q[arm], self.action.copy()
        count = round(seconds * self.env.control_hz)
        if count < 1:
            raise ValueError("Motion duration must include at least one control step.")
        if grip is None:
            grip = float(q[-1])
        target = None if point is None else np.asarray(point, dtype=float)
        end = before.copy()
        if target is None:
            end[arm_slice][-1] = grip
        elif self.last_point[arm] is None:
            q[:5] = self.ik[arm].solve(target, q, yaw=yaw)
            q[-1] = grip
            end[arm_slice] = q
        frames = []
        initial_yaw = self.last_yaw[arm]
        for index in range(count):
            fraction = (index + 1) / count
            blend = fraction ** 3 * (10 + fraction * (-15 + 6 * fraction))
            if target is not None and self.last_point[arm] is not None:
                point_now = self.last_point[arm] + (target - self.last_point[arm]) * blend
                yaw_now = (initial_yaw + (yaw - initial_yaw) * blend
                           if yaw is not None and initial_yaw is not None else yaw)
                q[:5] = self.ik[arm].solve(point_now, q, yaw=yaw_now)
                q[-1] = grip
                self.action[arm_slice] = q
            else:
                self.action = before + (end - before) * blend
            frames.append(self.action.copy())
        self.q[arm] = self.action[arm_slice].copy()
        if target is not None:
            self.last_point[arm], self.last_yaw[arm] = target, yaw
        return phase, np.asarray(frames)

    def home(self, arm: str, phase: str, *, seconds=3.0) -> tuple[str, np.ndarray]:
        arm_slice = slice(0, 6) if arm == "A" else slice(6, 12)
        before, end = self.action.copy(), self.action.copy()
        end[arm_slice] = HOME[arm_slice]
        count, frames = round(seconds * self.env.control_hz), []
        for index in range(count):
            fraction = (index + 1) / count
            blend = fraction ** 3 * (10 + fraction * (-15 + 6 * fraction))
            frames.append(before + (end - before) * blend)
        self.action = end
        self.q[arm] = end[arm_slice].copy()
        self.last_point[arm] = None
        self.last_yaw[arm] = None
        return phase, np.asarray(frames)


class FullTaskController:
    """Execute the canonical graph without simulator object poses or contacts."""

    def __init__(self, env: BimanualTableEnv, plan, *, recovery_mode: str = "adaptive",
                 recovery_memory: RecoveryMemory | None = None,
                 episode_id: str = "local-episode", memory_write: bool = True,
                 recovery_candidates: tuple[RecoveryCandidate, ...] | None = None):
        validate_full_plan(plan)
        if recovery_mode not in RECOVERY_MODES:
            raise ValueError(f"recovery_mode must be one of {RECOVERY_MODES}")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id is required")
        self.env, self.plan = env, plan
        self.recovery_mode, self.recovery_memory = recovery_mode, recovery_memory
        self.episode_id, self.memory_write = episode_id, memory_write
        self.active_step = plan.steps[0]
        self.completed_steps: list[int] = []
        self.events: list[dict] = []
        self.evidence = {
            "controller_inputs": ["overhead_rgb", "robot_joint_positions", "command_graph", "action_history"],
            "perception": "calibrated color fiducials in overhead RGB",
            "learned_policy": False,
            "object_state_used_for_targets": False,
            "object_attachment_used": False,
            "sequence": "DAG-ready execution; goal command starts drawer and mug in parallel",
        }
        self.done, self.failed_reason = False, None
        self.phase = "initialize_full_task"
        self._motion = _MotionProgram(env)
        self._segments: list[tuple[str, np.ndarray]] = []
        self._segment_index = self._tick = 0
        self._step_index = 0
        self._step_started = float(env.data.time)
        self._drawer_initial = None
        self._parallel_mug_step = None
        self._park_b_with_next_step = False
        self.recovery_attempts = 0
        self._pending_recovery: tuple[RecoveryCandidate, float, RecoveryContext] | None = None
        self._attempted_recoveries: set[tuple[str, str]] = set()
        self._recovery_candidates = recovery_candidates or (
            RecoveryCandidate("arm_B_direct_table_regrasp", RecoverabilityVerdict.RETRY, "B",
                              .95, 14.0, True, True, True, True, 1),
            RecoveryCandidate("arm_B_cautious_table_regrasp", RecoverabilityVerdict.RETRY, "B",
                              .90, 16.5, True, True, True, True, 1, extra_cost=.5),
        )
        self._supervisor = (AdaptiveRecoverySupervisor(recovery_memory, max_attempts=2)
                            if recovery_memory is not None else RecoverySupervisor(max_attempts=2))
        self.evidence["recovery"] = {
            "mode": recovery_mode,
            "candidate_registry": [
                {"name": candidate.name, "arm": candidate.arm,
                 "validation_success_rate": candidate.success_rate,
                 "validation_count": candidate.evidence_count,
                 "estimated_duration_s": candidate.duration_s,
                 "expected_cost": candidate.expected_cost}
                for candidate in self._recovery_candidates
            ],
            "persistent_memory": recovery_memory is not None,
            "memory_updates_enabled": bool(recovery_memory is not None and memory_write),
        }

    def _record_detection(self, name: str, detection, *, step_id: int | None = None) -> np.ndarray:
        record = asdict(detection)
        detected_step = self.active_step.id if step_id is None else step_id
        self.evidence.setdefault("detections", []).append({"step": detected_step, "object": name, **record})
        self.events.append({"type": "object_localized", "step_id": detected_step,
                            "object": name, "source": "overhead_rgb", **record})
        return np.asarray(detection.xy)

    def _mug_segments(self, xy: np.ndarray) -> list[tuple[str, np.ndarray]]:
        m = self._motion
        m.use_tcp("B", narrow=False)
        segments = [m.move("B", "approach_mug", [*xy, .94], grip=.8, seconds=3, yaw=np.pi / 2),
                    m.move("B", "descend_to_mug", [*xy, .83], grip=.8, yaw=np.pi / 2),
                    m.move("B", "grasp_mug", grip=-.1, seconds=1),
                    m.move("B", "lift_mug", [*xy, .94], grip=-.1, yaw=np.pi / 2),
                    m.move("B", "carry_mug_upper_right", [.25, .20, .94], grip=-.1, seconds=3, yaw=np.pi),
                    m.move("B", "lower_mug", [.25, .20, .83], grip=-.1, yaw=np.pi),
                    m.move("B", "release_mug", grip=.25, seconds=1),
                    m.move("B", "retract_from_mug", [.25, .20, .94], grip=.25, yaw=np.pi),
                    m.move("B", "open_after_mug", grip=.8, seconds=1),
                    m.home("B", "park_after_mug"),
                    m.move("B", "settle_full_task", grip=.8, seconds=1.2)]
        m.use_tcp("B", narrow=True)
        return segments

    @staticmethod
    def _parallel_segments(a_segments: list[tuple[str, np.ndarray]],
                           b_segments: list[tuple[str, np.ndarray]], *,
                           phase: str = "parallel_drawer_and_mug") -> list[tuple[str, np.ndarray]]:
        """Merge precomputed disjoint-arm trajectories at the control cadence."""

        a = np.concatenate([trajectory for _, trajectory in a_segments])
        b = np.concatenate([trajectory for _, trajectory in b_segments])
        count = max(len(a), len(b))
        merged = np.empty((count, 12), dtype=float)
        for index in range(count):
            merged[index, :6] = a[min(index, len(a) - 1), :6]
            merged[index, 6:] = b[min(index, len(b) - 1), 6:]
        return [(phase, merged)]

    def _prepare_step(self) -> None:
        step = self.active_step
        top = self.env.render("top")
        segments: list[tuple[str, np.ndarray]] = []
        m = self._motion
        if step.skill == "open_drawer":
            xy = self._record_detection("drawer_handle", locate_handle(top))
            self._drawer_initial = xy
            m.use_tcp("A", narrow=False)
            m.seed("A", [-.8, .7, -.4, 1, .2, .8])
            handle = np.r_[xy, .885]
            m.ik["A"].solve(handle, m.q["A"], yaw=None)
            axis = m.ik["A"].scratch.xmat[m.ik["A"].body].reshape(3, 3)[:, 0]
            aligned = handle + .006 * axis
            segments += [m.move("A", "approach_handle", handle + [0, 0, .055], grip=.8, seconds=3, yaw=None),
                         m.move("A", "descend_to_handle", handle, grip=.8, yaw=None),
                         m.move("A", "align_handle", aligned, grip=.8, seconds=1, yaw=None),
                         m.move("A", "grasp_handle", grip=-.1, seconds=1.5),
                         m.move("A", "pull_drawer", aligned + [-.165, 0, 0], grip=-.1, seconds=5, yaw=None),
                         m.move("A", "hold_drawer", grip=-.1, seconds=1),
                         m.move("A", "release_handle", grip=.8, seconds=1),
                         m.move("A", "retract_from_drawer", aligned + [-.165, 0, .055], grip=.8, yaw=None),
                         m.home("A", "park_after_drawer")]
            m.use_tcp("A", narrow=True)
            independent_mug = next((candidate for candidate in self.plan.steps
                                    if candidate.object == "mug" and not candidate.needs), None)
            if independent_mug is not None and independent_mug.id not in self.completed_steps:
                mug_xy = self._record_detection("mug", locate_full_mug(top), step_id=independent_mug.id)
                mug_segments = self._mug_segments(mug_xy)
                segments = self._parallel_segments(segments, mug_segments)
                self._parallel_mug_step = independent_mug
                self.events.append({"type": "parallel_group_started", "step_ids": [step.id, independent_mug.id],
                                    "arms": ["A", "B"], "geometry_check": "validated_full_scene_pair"})
        elif step.object == "plate":
            center = self._record_detection("plate", locate_plate(top))
            rim = center + [-.076, 0]
            m.use_tcp("A", narrow=True)
            segments += [m.move("A", "approach_plate_rim", [*rim, .91], grip=.22, seconds=3, yaw=0),
                         m.move("A", "descend_to_plate_rim", [*rim, .812], grip=.22, yaw=0),
                         m.move("A", "grasp_plate_rim", grip=-.17, seconds=1),
                         m.move("A", "lift_plate", [*rim, .91], grip=-.17, yaw=0),
                         m.move("A", "carry_plate_to_center", [-.026, 0, .91], grip=-.17, seconds=3, yaw=0),
                         m.move("A", "lower_plate", [-.026, 0, .812], grip=-.17, yaw=0),
                         m.move("A", "release_plate", grip=.22, seconds=1),
                         m.move("A", "retract_from_plate", [-.026, 0, .91], grip=.22, yaw=0),
                         m.home("A", "park_after_plate")]
        elif step.skill == "pick" and step.object == "spoon":
            xy = self._record_detection("spoon", locate_utensil(top, "spoon", in_drawer=True))
            m.use_tcp("A", narrow=True)
            segments += [m.move("A", "approach_spoon", [*xy, .91], grip=.22, seconds=3, yaw=np.pi / 2),
                         m.move("A", "descend_to_spoon", [*xy, .84], grip=.22, yaw=np.pi / 2),
                         m.move("A", "grasp_spoon", grip=-.17, seconds=1),
                         m.move("A", "raise_spoon_in_drawer", [*xy, .88], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "extract_spoon", [-.18, xy[1], .88], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "lift_spoon_clear", [-.18, xy[1], .95], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "carry_spoon_clearance", [-.10, .36, .95], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "carry_spoon_shared", [-.01, .30, .95], grip=-.17, seconds=3, yaw=np.pi / 2),
                         m.move("A", "stage_spoon_alignment", [0, .20, .95], grip=-.17, seconds=2, yaw=np.pi / 2),
                         m.move("A", "align_spoon_vertical", [0, .20, .95], grip=-.17, seconds=3, yaw=0),
                         m.move("A", "present_spoon", [.08, .18, .95], grip=-.17, seconds=3, yaw=0),
                         m.move("A", "lower_spoon_handoff", [.10, .18, .92], grip=-.17, seconds=3, yaw=0)]
        elif step.skill == "handoff":
            segments += [m.move("B", "receiver_staging", [.24, .16, .96], grip=.22, seconds=2, yaw=np.pi),
                         m.move("B", "receiver_approach", [.12, .12, .95], grip=.22, seconds=3, yaw=np.pi),
                         m.move("B", "receiver_align", [.10, .12, .883], grip=.22, seconds=2, yaw=np.pi),
                         m.move("B", "receiver_grasp", grip=-.174, seconds=1.5),
                         m.move("A", "donor_release", grip=.22, seconds=1),
                         m.move("A", "donor_retract", [.08, .18, .95], grip=.22, yaw=0),
                         m.move("A", "donor_clear", [0, .18, .94], grip=.22, yaw=0),
                         m.move("B", "receiver_retention", [.12, .12, .93], grip=-.174, seconds=2, yaw=np.pi)]
        elif step.object == "spoon" and step.arm == "B":
            # Arm B holds the lower handle after the vertical handoff, so the
            # TCP target compensates for the spoon center's measured offset.
            segments += [m.move("B", "carry_spoon_right", [.24, -.05, .93], grip=-.174, seconds=4, yaw=np.pi),
                         m.move("B", "lower_spoon_right", [.24, -.05, .82], grip=-.174, yaw=np.pi),
                         m.move("B", "release_spoon", grip=.22, seconds=1),
                         # Retracting clears the overhead view enough to verify the
                         # released spoon.  Parking happens only after verification,
                         # so a miss can be recovered without a needless round trip.
                         m.move("B", "retract_from_spoon", [.24, -.05, .93], grip=.22, yaw=np.pi),
                         m.move("B", "clear_spoon_camera", [.24, .16, .96], grip=.22,
                                seconds=1, yaw=np.pi),
                         m.move("B", "settle_spoon_before_verify", grip=.22, seconds=1, yaw=None)]
        elif step.object == "fork":
            xy = self._record_detection("fork", locate_utensil(top, "fork", in_drawer=True))
            # The vivid handle centroid is already about 1 cm behind the body's
            # inertial origin, which is the successful grasp offset.
            grasp = xy
            segments += [m.move("A", "approach_fork", [*grasp, .95], grip=.22, seconds=3, yaw=np.pi / 2),
                         m.move("A", "descend_to_fork", [*grasp, .84], grip=.22, yaw=np.pi / 2),
                         m.move("A", "grasp_fork", grip=-.17, seconds=1),
                         m.move("A", "raise_fork_in_drawer", [*grasp, .88], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "extract_fork", [-.18, grasp[1], .88], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "lift_fork_clear", [-.18, grasp[1], .95], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "carry_fork_clearance", [-.10, .31, .95], grip=-.17, yaw=np.pi / 2),
                         m.move("A", "carry_fork_to_table", [0, .20, .94], grip=-.17, seconds=3, yaw=np.pi / 2),
                         m.move("A", "rotate_fork", [0, .10, .93], grip=-.17, seconds=3, yaw=0),
                         m.move("A", "carry_fork_left", [-.15, .02, .91], grip=-.17, seconds=3, yaw=0),
                         m.move("A", "lower_fork_left", [-.15, .02, .82], grip=-.17, yaw=0),
                         m.move("A", "release_fork", grip=.22, seconds=1),
                         m.move("A", "retract_from_fork", [-.15, .02, .91], grip=.22, seconds=1, yaw=0),
                         m.home("A", "park_after_fork", seconds=2)]
        elif step.object == "mug":
            xy = self._record_detection("mug", locate_full_mug(top))
            segments += self._mug_segments(xy)
        else:
            raise ValueError(f"No registered trajectory for plan step {step.id}.")
        if self._park_b_with_next_step:
            segments = self._parallel_segments(
                segments,
                [m.home("B", "park_after_verified_spoon", seconds=1.5)],
                phase="parallel_fork_and_arm_B_park",
            )
            self._park_b_with_next_step = False
            self.events.append({"type": "parallel_cleanup_started", "arm": "B",
                                "with_step": step.id, "reason": "verified_spoon"})
        self._segments = segments
        self._segment_index = self._tick = 0

    def _spoon_recovery(self, xy: np.ndarray, candidate: RecoveryCandidate) -> None:
        """Schedule the selected camera-grounded correction without a reset."""

        cautious = candidate.name == "arm_B_cautious_table_regrasp"
        self.events.extend((
            {"type": "recovery_selected", "action": "arm_B_table_regrasp",
             "candidate": candidate.name, "selection": "lowest-cost validated correction",
             "expected_cost": candidate.expected_cost, "attempt": self.recovery_attempts,
             "max_attempts": 2},
        ))
        self.evidence.setdefault("recoveries", []).append({
            "step": self.active_step.id,
            "failure_kind": "spoon_outside_goal",
            "action": candidate.name,
            "attempt": self.recovery_attempts,
            "reset_used": False,
            "outcome": "pending",
        })
        m = self._motion
        m.use_tcp("B", narrow=True)
        hover = .92 if cautious else .90
        grasp_height = .802 if cautious else .805
        close_seconds = 1.5 if cautious else 1.0
        self._segments = [
            m.move("B", "recovery_approach_spoon", [*xy, hover], grip=.22,
                   seconds=2.5 if cautious else 2, yaw=np.pi),
            m.move("B", "recovery_descend_spoon", [*xy, grasp_height], grip=.22,
                   seconds=2 if cautious else 1.5, yaw=np.pi),
            m.move("B", "recovery_regrasp_spoon", grip=-.17, seconds=close_seconds),
            m.move("B", "recovery_lift_spoon", [*xy, hover], grip=-.17,
                   seconds=2 if cautious else 1.5, yaw=np.pi),
            m.move("B", "recovery_carry_spoon", [.25, .004, hover], grip=-.17,
                   seconds=3 if cautious else 2.5, yaw=np.pi),
            m.move("B", "recovery_lower_spoon", [.25, .004, .82], grip=-.17, seconds=1.5, yaw=np.pi),
            m.move("B", "recovery_release_spoon", grip=.22, seconds=1),
            m.move("B", "recovery_retract_spoon", [.25, .004, hover], grip=.22, seconds=1, yaw=np.pi),
            m.move("B", "recovery_clear_spoon_camera", [.24, .16, .96], grip=.22,
                   seconds=1, yaw=np.pi),
            m.move("B", "recovery_settle_spoon_before_verify", grip=.22, seconds=1, yaw=None),
        ]
        self._segment_index = self._tick = 0

    def _blind_spoon_retry(self) -> None:
        """Repeat the failed place command without failure-specific localization."""

        self.recovery_attempts += 1
        self.events.append({"type": "retry_started", "action": "repeat_spoon_place",
                            "attempt": self.recovery_attempts, "max_attempts": 1})
        self.evidence.setdefault("recoveries", []).append({
            "step": self.active_step.id, "failure_kind": "spoon_outside_goal",
            "action": "repeat_spoon_place", "attempt": self.recovery_attempts,
            "reset_used": False, "outcome": "pending",
        })
        m = self._motion
        self._segments = [
            m.move("B", "retry_close_gripper", grip=-.17, seconds=1),
            m.move("B", "retry_receiver_staging", [.24, .08, .96], grip=-.17,
                   seconds=2, yaw=np.pi / 2),
            m.move("B", "retry_carry_spoon_right", [.30, .004, .93], grip=-.17, seconds=3, yaw=np.pi / 2),
            m.move("B", "retry_lower_spoon_right", [.30, .004, .82], grip=-.17, seconds=1.5, yaw=np.pi / 2),
            m.move("B", "retry_release_spoon", grip=.22, seconds=1),
            m.move("B", "retry_retract", [.30, .004, .93], grip=.22, seconds=1, yaw=np.pi / 2),
            m.move("B", "retry_clear_spoon_camera", [.24, .16, .96], grip=.22,
                   seconds=1, yaw=np.pi),
            m.move("B", "retry_settle_spoon_before_verify", grip=.22, seconds=1, yaw=None),
        ]
        self._segment_index = self._tick = 0

    def _recovery_context(self, xy: np.ndarray) -> RecoveryContext:
        bucket = f"table_xy_{round(float(xy[0]) / .02)}_{round(float(xy[1]) / .02)}"
        return RecoveryContext("pick_place", "spoon", "verify_goal", "outside_goal",
                               bucket, "mise_full_scene_v1", "deterministic_contact_v2")

    def _finish_pending_recovery(self, outcome: str) -> None:
        if self._pending_recovery is None:
            return
        candidate, started, context = self._pending_recovery
        duration = max(.001, float(self.env.data.time) - started)
        if self.recovery_memory is not None and self.memory_write:
            self.recovery_memory.record(
                context, candidate,
                attempt_id=f"{self.episode_id}:step{self.active_step.id}:attempt{self.recovery_attempts}",
                episode_id=self.episode_id, outcome=outcome, duration_s=duration,
                observation_ref=f"trace:{self.episode_id}:spoon-verify:{self.recovery_attempts}",
            )
        recoveries = self.evidence.get("recoveries", [])
        if recoveries:
            recoveries[-1].update(outcome=outcome, duration_s=duration)
        self.events.append({"type": f"recovery_{'succeeded' if outcome == 'success' else 'failed'}",
                            "candidate": candidate.name, "attempt": self.recovery_attempts,
                            "duration_s": duration, "memory_updated": bool(self.recovery_memory and self.memory_write)})
        self._pending_recovery = None

    def _handle_spoon_failure(self, xy: np.ndarray) -> bool:
        self.events.append({"type": "failure_detected", "failure_kind": "spoon_outside_goal",
                            "source": "overhead_rgb", "attempt": self.recovery_attempts + 1})
        if self.recovery_mode == "none":
            raise ValueError("RGB verification found the spoon outside its goal; recovery is disabled.")
        if self.recovery_mode == "blind_retry":
            if self.recovery_attempts >= 1:
                if self.evidence.get("recoveries"):
                    self.evidence["recoveries"][-1]["outcome"] = "failure"
                self.events.append({"type": "retry_failed", "action": "repeat_spoon_place",
                                    "attempt": self.recovery_attempts})
                raise ValueError("RGB verification found the spoon outside its goal after bounded retry.")
            self._blind_spoon_retry()
            return False

        if self._pending_recovery is not None:
            self._finish_pending_recovery("failure")
        context = self._recovery_context(xy)
        evidence = MonitorEvidence(failure_detected=True, fresh=True, stable_for_replan=True)
        candidates = tuple(candidate for candidate in self._recovery_candidates
                           if (candidate.name, candidate.arm) not in self._attempted_recoveries)
        if isinstance(self._supervisor, AdaptiveRecoverySupervisor):
            decision = self._supervisor.decide_with_memory(
                self.active_step.id, evidence, candidates, context=context,
                episode_id=self.episode_id, elapsed_s=float(self.env.data.time), required_arm="B")
        else:
            decision = self._supervisor.decide(
                self.active_step.id, evidence, candidates,
                elapsed_s=float(self.env.data.time), required_arm="B")
        candidate = self._supervisor.selected
        if candidate is None or decision.verdict not in {RecoverabilityVerdict.RETRY, RecoverabilityVerdict.REPLAN}:
            raise ValueError(f"Recovery supervisor stopped: {decision.reason}")
        self._supervisor.begin_recovery(self.active_step.id)
        self.recovery_attempts += 1
        self._attempted_recoveries.add((candidate.name, candidate.arm))
        self._pending_recovery = (candidate, float(self.env.data.time), context)
        self._spoon_recovery(xy, candidate)
        return False

    def _verify_step(self) -> bool:
        step, top = self.active_step, self.env.render("top")
        verification: dict = {"step": step.id, "source": "overhead_rgb"}
        if step.skill == "open_drawer":
            final = locate_handle(top, opened=True)
            error = float(self._drawer_initial[0] - final.xy[0])
            if error < .144:
                raise ValueError("RGB verification found insufficient drawer travel.")
            verification.update(object="drawer_handle", visual_travel_m=error, detection=asdict(final))
            if self._parallel_mug_step is not None:
                mug = locate_full_mug(top)
                mug_error = float(np.linalg.norm(np.asarray(mug.xy) - [.25, .20]))
                if mug_error > .02:
                    raise ValueError("RGB verification found the parallel mug outside its upper-right goal.")
                mug_verification = {"step": self._parallel_mug_step.id, "source": "overhead_rgb",
                                    "object": "mug", "goal_error_m": mug_error,
                                    "detection": asdict(mug), "parallel_with": step.id}
                self.evidence.setdefault("visual_verifications", []).append(mug_verification)
                self.events.append({"type": "visual_step_verified", **mug_verification})
        elif step.object == "plate":
            final = locate_plate(top); error = float(np.linalg.norm(np.asarray(final.xy) - [.05, 0]))
            if error > .02:
                raise ValueError("RGB verification found the plate outside its center goal.")
            verification.update(object="plate", goal_error_m=error, detection=asdict(final))
        elif step.skill == "pick" or step.skill == "handoff":
            # Acquisition provenance is scored by the isolated contact evaluator.
            # The controller records its RGB grounding and commanded transition.
            verification.update(object="spoon", transition="presented" if step.skill == "pick" else "A_to_B_commanded")
        elif step.object in {"spoon", "fork"}:
            final = locate_utensil(top, step.object, in_drawer=False)
            goal = [.25, 0] if step.object == "spoon" else [-.15, 0]
            error = float(np.linalg.norm(np.asarray(final.xy) - goal))
            if error > .025:
                if step.object == "spoon":
                    return self._handle_spoon_failure(np.asarray(final.xy))
                raise ValueError(f"RGB verification found the {step.object} outside its goal.")
            if final.axis_error_degrees is None or final.axis_error_degrees > 20:
                if step.object == "spoon":
                    return self._handle_spoon_failure(np.asarray(final.xy))
                raise ValueError("RGB verification found the fork outside its vertical alignment tolerance.")
            verification.update(object=step.object, goal_error_m=error,
                                axis_error_degrees=final.axis_error_degrees,
                                detection=asdict(final))
        elif step.object == "mug":
            final = locate_full_mug(top); error = float(np.linalg.norm(np.asarray(final.xy) - [.25, .20]))
            if error > .02:
                raise ValueError("RGB verification found the mug outside its upper-right goal.")
            verification.update(object="mug", goal_error_m=error, detection=asdict(final))
        self.evidence.setdefault("visual_verifications", []).append(verification)
        self.events.append({"type": "visual_step_verified", **verification})
        if step.object == "spoon" and self._pending_recovery is not None:
            self._finish_pending_recovery("success")
        return True

    def advance(self) -> np.ndarray:
        if self.done or self.failed_reason:
            return self._motion.action.copy()
        try:
            if not self._segments:
                self._prepare_step()
            if float(self.env.data.time) - self._step_started > self.active_step.timeout_s:
                raise ValueError(f"Step {self.active_step.id} exceeded its instruction timeout.")
            if self._segment_index == len(self._segments):
                if not self._verify_step():
                    return self.advance()
                # Arm B is already in a nearby camera-clear pose.  Once the spoon
                # is verified, park B concurrently with arm A's independent fork
                # motion instead of adding another serial task segment.
                if self.active_step.object == "spoon" and self.active_step.arm == "B":
                    self._park_b_with_next_step = True
                self.completed_steps.append(self.active_step.id)
                if self._parallel_mug_step is not None and self.active_step.skill == "open_drawer":
                    self.completed_steps.append(self._parallel_mug_step.id)
                    self.events.append({"type": "parallel_group_completed",
                                        "step_ids": [self.active_step.id, self._parallel_mug_step.id],
                                        "arms": ["A", "B"]})
                    self._parallel_mug_step = None
                self._step_index += 1
                while (self._step_index < len(self.plan.steps)
                       and self.plan.steps[self._step_index].id in self.completed_steps):
                    self._step_index += 1
                if self._step_index == len(self.plan.steps):
                    self.active_step = None
                    self.done, self.phase = True, "completed"
                    return self._motion.action.copy()
                self.active_step = self.plan.steps[self._step_index]
                self._segments = []
                self._step_started = float(self.env.data.time)
                return self.advance()
            self.phase, trajectory = self._segments[self._segment_index]
            self._motion.action = trajectory[self._tick].copy()
            self._tick += 1
            if self._tick == len(trajectory):
                self._segment_index += 1
                self._tick = 0
            return self._motion.action.copy()
        except ValueError as exc:
            self.failed_reason, self.phase = str(exc), "failed"
            self.events.append({"type": "skill_failed", "reason": str(exc)})
            return self._motion.action.copy()


class FullTaskEvaluator:
    """Adapter exposing the independent full evaluator to the worker."""

    full_task = True

    def __init__(self, env):
        self.evaluator = TaskEvaluator(env.model, config_path=FULL_CONFIG)
        self.snapshot = None
        self.success = False
        self.evidence = {}

    def update(self, data):
        self.snapshot = self.evaluator.update(data)
        self.success = self.snapshot.full_task_success
        self.evidence = dict(self.snapshot.evidence)
        return self.success
