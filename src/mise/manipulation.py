"""Camera-guided contact baseline, producing only twelve robot joint targets.

This is a deterministic skill expert, not ACT or a general language model.
Privileged object state is never read here. Success certification lives in the
separate contact evaluator, which is not an input to this controller.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .contact_vision import locate_mug
from .kinematics import ArmKinematics
from .planner import validate_plan_graph
from .sim import BimanualTableEnv, ROOT
from .telemetry import CONTACT_COMMAND
from .types import Plan

SUPPORTED_CONTACT_COMMANDS = (CONTACT_COMMAND,)
HOME = np.array([0, -.5, .7, -.3, 0, .8, -1.3, -.5, .7, -.3, 0, .8], dtype=float)


def validate_contact_plan(plan: Plan) -> None:
    validate_plan_graph(plan)
    if len(plan.steps) != 1 or any((s.skill, s.arm, s.object, s.target) != ('pick_place', 'B', 'mug', 'table_upper_right') or s.needs for s in plan.steps):
        raise ValueError('The contact expert currently supports: ' + CONTACT_COMMAND + ' Drawer grasping, other objects, arm A and hand-off are still under development.')


def create_contact_env(seed: int = 0, *, scene_path: Path | None = None) -> BimanualTableEnv:
    from runpy import run_path
    build_contact_scene = run_path(str(ROOT / 'scripts/build_contact_scene.py'))['build_contact_scene']
    env = BimanualTableEnv(seed=seed, scene_path=scene_path or build_contact_scene(), randomize=False, initial_observation=False)
    # Initial-state sampling only. Once returned, execution writes robot controls.
    env.data.qpos[env._arm_qpos.ravel()] = HOME
    env.data.ctrl[env._arm_ctrl.ravel()] = HOME
    address = env.model.joint('mug_free').qposadr[0]
    env.data.qpos[address:address + 2] += np.random.default_rng(seed).uniform(-.008, .008, size=2)
    env._mj.mj_forward(env.model, env.data)
    for _ in range(300):
        env._mj.mj_step(env.model, env.data)
    return env


class ManipulationController:
    """Execute a validated skill with camera grounding and proprioceptive targets."""
    def __init__(self, env: BimanualTableEnv, plan: Plan):
        validate_contact_plan(plan)
        self.env, self.plan = env, plan
        self.active_step = plan.steps[0]
        self.completed_steps = []
        self.events = []
        self.evidence = {'controller_inputs': ['overhead_rgb', 'robot_joint_positions'],
                         'learned_policy': False, 'object_state_used_for_targets': False}
        self.done, self.failed_reason = False, None
        self.phase = 'locate_mug'
        self._ik = ArmKinematics(env.model, 'B')
        self._action = env.joint_positions()
        self._segments = []
        self._index, self._tick = 0, 0
        self._started = float(env.data.time)

    def _prepare(self):
        detection = locate_mug(self.env.render('top'))
        xy = np.asarray(detection.xy)
        if not (.065 < xy[0] < .14 and -.01 < xy[1] < .05):
            raise ValueError('Detected mug is outside the validated pickup area.')
        self.evidence['initial_detection'] = asdict(detection)
        self.events.append({'type': 'object_localized', 'object': 'mug', 'source': 'overhead_rgb', **asdict(detection)})
        # Warm-start successive IK targets. This known robot configuration fixes
        # the initial free wrist orientation; pickup XY comes from the image.
        q = np.array([0, -.5, .7, -.3, 0, .8])
        previous_target = None
        previous_action = self._action.copy()
        for phase, target, grip, seconds in [
            ('approach_mug', [*xy, .94], .8, 2.5),
            ('descend_to_grasp', [*xy, .83], .8, 2.),
            ('close_gripper', None, -.1, 1.),
            ('lift_mug', [*xy, .94], -.1, 2.),
            ('carry_clearance', [min(float(xy[0]), .10), .16, .94], -.1, 2.),
            ('carry_to_goal', [.25, .20, .94], -.1, 2.5),
            ('lower_mug', [.25, .20, .83], -.1, 2.),
            ('release_mug', None, .8, 1.),
            ('retract_gripper', [.25, .20, .94], .8, 2.),
            ('clear_camera', None, .8, 2.5),
            ('settle_and_verify', None, .8, .8),
        ]:
            count = round(seconds * self.env.control_hz)
            trajectory = []
            if target is not None and previous_target is not None:
                # Cartesian interpolation prevents a joint-space arc from
                # sweeping the fixed finger through the object's top surface.
                for i in range(count):
                    t = (i + 1) / count
                    blend = t ** 3 * (10 + t * (-15 + 6 * t))
                    point = np.asarray(previous_target) + (np.asarray(target) - previous_target) * blend
                    q[:5] = self._ik.solve(point, q)
                    q[5] = grip
                    action = HOME.copy()
                    action[6:] = q
                    trajectory.append(action)
            else:
                if target is not None:
                    q[:5] = self._ik.solve(target, q)
                q[5] = grip
                if phase == 'clear_camera':
                    q = HOME[6:].copy()
                end_action = HOME.copy()
                end_action[6:] = q
                for i in range(count):
                    t = (i + 1) / count
                    blend = t ** 3 * (10 + t * (-15 + 6 * t))
                    trajectory.append(previous_action + (end_action - previous_action) * blend)
            if target is not None:
                previous_target = np.asarray(target)
            previous_action = trajectory[-1].copy()
            self._segments.append((phase, np.asarray(trajectory)))

    def advance(self) -> np.ndarray:
        if self.done or self.failed_reason:
            return self._action.copy()
        try:
            if not self._segments:
                self._prepare()
            if float(self.env.data.time) - self._started > self.active_step.timeout_s:
                raise ValueError('Contact skill exceeded its instruction timeout.')
            if self._index == len(self._segments):
                detection = locate_mug(self.env.render('top'))
                error = float(np.linalg.norm(np.asarray(detection.xy) - [.25, .20]))
                self.evidence.update(final_detection=asdict(detection), camera_goal_error_m=error)
                if error > .02:
                    raise ValueError(f'Camera check found the mug {error * 100:.1f} cm from the goal; placement failed.')
                self.events.append({'type': 'visual_placement_verified', 'goal_error_m': error})
                self.completed_steps.append(self.active_step.id)
                self.active_step = None
                self.done, self.phase = True, 'completed'
                return self._action.copy()
            phase, trajectory = self._segments[self._index]
            self.phase = phase
            self._action = trajectory[self._tick].copy()
            self._tick += 1
            if self._tick >= len(trajectory):
                self._index += 1
                self._tick = 0
            return self._action.copy()
        except ValueError as exc:
            self.failed_reason, self.phase = str(exc), 'failed'
            self.events.append({'type': 'skill_failed', 'reason': str(exc)})
            return self._action.copy()
