"""Numerical robot-only IK. The scratch state never changes the live simulator."""
from __future__ import annotations
import numpy as np
from .sim import ARM_JOINTS


class ArmKinematics:
    def __init__(self, model, arm: str):
        import mujoco
        self.mj, self.model = mujoco, model
        names = [f'arm_{arm.lower()}_{name}' for name in ARM_JOINTS[:5]]
        joints = [model.joint(name) for name in names]
        self.qpos = np.array([j.qposadr[0] for j in joints])
        self.dofs = np.array([j.dofadr[0] for j in joints])
        self.limits = np.array([j.range for j in joints])
        self.site = model.site(f'arm_{arm.lower()}_tcp').id
        self.body = model.body(f'arm_{arm.lower()}_gripper').id
        self.scratch = mujoco.MjData(model)

    def solve(self, target, start, *, yaw: float | None = None):
        """Position plus downward approach axis; free rotation about the tool axis."""
        data, model = self.scratch, self.model
        data.qpos[self.qpos] = np.asarray(start)[:5]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(300):
            self.mj.mj_kinematics(model, data)
            self.mj.mj_comPos(model, data)
            rotation = data.xmat[self.body].reshape(3, 3)
            z = rotation[:, 2]
            if yaw is None:
                angular_error = np.cross(z, [0, 0, 1])
            else:
                desired = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                angular_error = sum(np.cross(rotation[:, i], desired[:, i]) for i in range(3)) / 2
            error = np.r_[np.asarray(target) - data.site_xpos[self.site], .14 * angular_error]
            if np.linalg.norm(error) < .0003 and z[2] > .99:
                return data.qpos[self.qpos].copy()
            self.mj.mj_jacSite(model, data, jp, jr, self.site)
            angular_jac = (np.eye(3) - np.outer(z, z)) @ jr[:, self.dofs] if yaw is None else jr[:, self.dofs]
            jac = np.vstack([jp[:, self.dofs], .14 * angular_jac])
            change = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * .00005, error)
            data.qpos[self.qpos] = np.clip(data.qpos[self.qpos] + np.clip(change, -.15, .15), self.limits[:, 0] + .005, self.limits[:, 1] - .005)
        raise ValueError(f'No reachable downward grasp pose at {np.round(target, 3).tolist()}')
