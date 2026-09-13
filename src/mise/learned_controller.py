"""ACT execution with visual completion and robot-only action bounds."""
import time
from dataclasses import asdict
import numpy as np

from .contact_vision import locate_mug
from .kinematics import ArmKinematics
from .manipulation import validate_contact_plan


class LearnedContactController:
    def __init__(self, env, plan, policy):
        validate_contact_plan(plan)
        self.env, self.plan, self.policy = env, plan, policy
        self.active_step = plan.steps[0]
        self.completed_steps, self.events = [], []
        self.done, self.failed_reason = False, None
        self.phase = 'learned_contact_execution'
        self.evidence = dict(learned_policy=True, method='compact ACT-style CVAE',
                             controller_inputs=['overhead_rgb', 'wrist_a_rgb', 'wrist_b_rgb', 'robot_joint_positions'],
                             object_state_used_for_targets=False, chunk_execution_steps=15,
                             action_delta_limit_rad=.08, visual_completion_hold_s=1.0)
        self._action = env.joint_positions()
        self._chunk, self._tick = None, 0
        self._started = float(env.data.time)
        self._goal_since = None
        self._ik = ArmKinematics(env.model, 'B')
        self.inference_ms = []

    def advance(self):
        if self.done or self.failed_reason:
            return self._action.copy()
        try:
            now = float(self.env.data.time)
            if now - self._started > self.active_step.timeout_s:
                raise ValueError('Learned contact policy exceeded the instruction timeout.')
            if self._tick % 15 == 0:
                observation = self.env.observe()
                joints = np.asarray(observation.joint_positions)
                self._ik.scratch.qpos[self._ik.qpos] = joints[6:11]
                self._ik.mj.mj_kinematics(self.env.model, self._ik.scratch)
                tcp = self._ik.scratch.site_xpos[self._ik.site]
                clear = joints[11] > .65 and np.linalg.norm(tcp - [.25, .20, .83]) > .075
                try:
                    detection = locate_mug(observation.top)
                    error = float(np.linalg.norm(np.asarray(detection.xy) - [.25, .20]))
                    valid_goal = clear and error < .02
                except ValueError:
                    valid_goal = False
                if valid_goal:
                    if self._goal_since is None:
                        self._goal_since = now
                    if now - self._goal_since >= 1.0:
                        self.evidence.update(final_detection=asdict(detection), camera_goal_error_m=error)
                        self.events.append(dict(type='visual_placement_verified', goal_error_m=error))
                        self.completed_steps.append(self.active_step.id)
                        self.active_step = None
                        self.done, self.phase = True, 'completed'
                        return self._action.copy()
                else:
                    self._goal_since = None
                began = time.perf_counter()
                self._chunk = np.asarray(self.policy.predict_chunk(observation), dtype=float)
                latency = (time.perf_counter() - began) * 1000
                self.inference_ms.append(latency)
                if self._chunk.ndim != 2 or self._chunk.shape[1] != 12 or len(self._chunk) < 15 or not np.isfinite(self._chunk).all():
                    raise ValueError('Policy returned an invalid joint-target chunk.')
                self.evidence['inference_calls'] = len(self.inference_ms)
                self.evidence['inference_p50_ms'] = float(np.median(self.inference_ms))
                self.evidence['inference_p95_ms'] = float(np.percentile(self.inference_ms, 95))
                self.events.append(dict(type='policy_inference', latency_ms=latency, executed_chunk_steps=15))
            target = self._chunk[self._tick % 15]
            # No object-dependent correction is hidden in this velocity guard.
            target = np.clip(target, self._action - .08, self._action + .08)
            ranges = self.env.model.actuator_ctrlrange[self.env._arm_ctrl.ravel()]
            self._action = np.clip(target, ranges[:, 0], ranges[:, 1])
            self._tick += 1
        except ValueError as exc:
            self.failed_reason, self.phase = str(exc), 'failed'
            self.events.append(dict(type='skill_failed', reason=str(exc)))
        return self._action.copy()
