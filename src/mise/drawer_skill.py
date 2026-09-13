"""RGB-guided physical drawer opening; never drives the drawer joint."""
from dataclasses import asdict
from runpy import run_path

import numpy as np

from .contact_vision import Detection
from .kinematics import ArmKinematics
from .manipulation import HOME as MUG_HOME
from .sim import BimanualTableEnv, ROOT

DRAWER_COMMAND = 'Open the drawer with arm A.'
HOME = MUG_HOME.copy()
HOME[0] = 1.5  # Park arm A clear of the knob in the overhead image.


def locate_handle(rgb):
    image = np.asarray(rgb, dtype=float)
    height, width = image.shape[:2]
    scale = 2 * (1.75 - .89) * np.tan(np.deg2rad(29)) / height
    rows, cols = np.indices((height, width))
    world_y = ((height - 1) / 2 - rows) * scale
    world_x = (cols - (width - 1) / 2) * scale
    mask = ((image[:, :, 0] > 100) & (image[:, :, 1] > .55 * image[:, :, 0])
            & (image[:, :, 2] < .35 * image[:, :, 0])
            & (world_y > .28) & (world_y < .37) & (world_x > -.28) & (world_x < .02))
    points = np.argwhere(mask)
    if len(points) < 30:
        raise ValueError('The yellow drawer knob is not sufficiently visible.')
    row, col = np.mean(points, axis=0)
    return Detection(((float(col) - (width - 1) / 2) * scale,
                      ((height - 1) / 2 - float(row)) * scale),
                     len(points), (float(col), float(row)))


def create_drawer_env(seed=0):
    scene = run_path(str(ROOT / 'scripts/build_drawer_scene.py'))['build_drawer_scene']()
    env = BimanualTableEnv(seed=seed, scene_path=scene, randomize=False, initial_observation=False)
    env.data.qpos[env._arm_qpos.ravel()] = HOME
    env.data.ctrl[env._arm_ctrl.ravel()] = HOME
    env._mj.mj_forward(env.model, env.data)
    for _ in range(300):
        env._mj.mj_step(env.model, env.data)
    return env


class DrawerContactController:
    def __init__(self, env, plan):
        self.env, self.plan = env, plan
        self.active_step = plan.steps[0]
        self.completed_steps, self.events = [], []
        self.done, self.failed_reason = False, None
        self.phase = 'locate_drawer_handle'
        self.evidence = dict(controller_inputs=['overhead_rgb', 'robot_joint_positions'],
                             learned_policy=False, object_state_used_for_targets=False,
                             supported_skill='open_drawer', scene_randomization='none')
        self._action = env.joint_positions()
        self._segments, self._index, self._tick = [], 0, 0
        self._started = float(env.data.time)

    def _prepare(self):
        detection = locate_handle(self.env.render('top'))
        self._initial = np.asarray(detection.xy)
        if not (-.065 < self._initial[0] < -.035 and .31 < self._initial[1] < .34):
            raise ValueError('Drawer handle lies outside the validated closed-drawer region.')
        self.evidence['initial_detection'] = asdict(detection)
        self.events.append(dict(type='object_localized', object='drawer_handle',
                                source='overhead_rgb', **asdict(detection)))
        ik = ArmKinematics(self.env.model, 'A')
        q = np.array([-.8, .7, -.4, 1, .2, .8])
        handle = np.r_[self._initial, .885]
        q[:5] = ik.solve(handle, q)
        # Rigid knob alignment is computed from robot FK, not object state.
        aligned = handle + .006 * ik.scratch.xmat[ik.body].reshape(3, 3)[:, 0]
        stages = [('approach_handle', handle + [0, 0, .055], .8, 3),
                  ('descend_to_handle', handle, .8, 2),
                  ('align_handle', aligned, .8, 1),
                  ('grasp_handle', None, -.1, 1.5),
                  ('pull_drawer', aligned + [-.165, 0, 0], -.1, 5),
                  ('hold_drawer', None, -.1, 1),
                  ('release_handle', None, .8, 1),
                  ('retract_from_drawer', aligned + [-.165, 0, .055], .8, 2),
                  ('clear_camera', None, .8, 3),
                  ('verify_drawer', None, .8, 1)]
        previous_point = None
        previous = self._action.copy()
        q = np.array([-.8, .7, -.4, 1, .2, .8])
        for phase, target, grip, seconds in stages:
            count = round(seconds * self.env.control_hz)
            frames = []
            end = previous.copy()
            if target is None or previous_point is None:
                if target is not None:
                    q[:5] = ik.solve(target, q)
                q[5] = grip
                if phase == 'clear_camera':
                    q = HOME[:6].copy()
                end[:6] = q
            for tick in range(count):
                t = (tick + 1) / count
                blend = t ** 3 * (10 + t * (-15 + 6 * t))
                if target is not None and previous_point is not None:
                    point = previous_point + (target - previous_point) * blend
                    q[:5] = ik.solve(point, q)
                    q[5] = grip
                    action = HOME.copy(); action[:6] = q
                else:
                    action = previous + (end - previous) * blend
                frames.append(action.copy())
            if target is not None:
                previous_point = target.copy()
            previous = frames[-1].copy()
            self._segments.append((phase, frames))

    def advance(self):
        if self.done or self.failed_reason:
            return self._action.copy()
        try:
            if not self._segments:
                self._prepare()
            if float(self.env.data.time) - self._started > self.active_step.timeout_s:
                raise ValueError('Physical drawer skill exceeded its instruction timeout.')
            if self._index == len(self._segments):
                final = locate_handle(self.env.render('top'))
                travel = self._initial[0] - final.xy[0]
                self.evidence.update(final_detection=asdict(final), visual_drawer_travel_m=float(travel))
                if travel < .144 or abs(final.xy[1] - self._initial[1]) > .02:
                    raise ValueError('Camera check did not confirm sufficient drawer opening.')
                self.events.append(dict(type='visual_drawer_verified', travel_m=float(travel)))
                self.completed_steps.append(self.active_step.id)
                self.active_step = None
                self.done, self.phase = True, 'completed'
            else:
                self.phase, trajectory = self._segments[self._index]
                self._action = trajectory[self._tick].copy()
                self._tick += 1
                if self._tick == len(trajectory):
                    self._index += 1; self._tick = 0
        except ValueError as exc:
            self.failed_reason, self.phase = str(exc), 'failed'
            self.events.append(dict(type='skill_failed', reason=str(exc)))
        return self._action.copy()
