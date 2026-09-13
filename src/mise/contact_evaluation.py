"""Read-only physical outcome evidence, isolated from camera-guided control."""
import numpy as np
from .evaluation import StablePredicate


class ContactSkillEvaluator:
    def __init__(self, model):
        import mujoco
        self.mj, self.model = mujoco, model
        self.body = model.body('mug').id
        self.dof = model.joint('mug_free').dofadr[0]
        self.pads = {model.geom(f'arm_b_{pad}_pad').id for pad in ('fixed', 'moving')}
        self.table = model.geom('table_top').id
        self.lift = StablePredicate(.2, .1)
        self.place = StablePredicate(1., .1)
        self.held = False
        self.lifted = False
        self.success = False
        self.max_height = .815
        self.evidence = {}

    def update(self, data):
        object_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.body))
        touched = set()
        for i, contact in enumerate(data.contact):
            if contact.geom1 not in object_geoms and contact.geom2 not in object_geoms:
                continue
            force = np.zeros(6)
            self.mj.mj_contactForce(self.model, data, i, force)
            if force[0] > .005:
                touched.add(contact.geom2 if contact.geom1 in object_geoms else contact.geom1)
        p = data.xpos[self.body].copy()
        bilateral = self.pads <= touched
        self.held |= bilateral
        self.max_height = max(self.max_height, float(p[2]))
        self.lifted |= self.lift.update(float(data.time), bilateral and p[2] > .845 and self.table not in touched)
        upright = float(data.xmat[self.body].reshape(3, 3)[2, 2]) > np.cos(np.deg2rad(15))
        speed = float(np.linalg.norm(data.qvel[self.dof:self.dof + 3]))
        angular_speed = float(np.linalg.norm(data.qvel[self.dof + 3:self.dof + 6]))
        goal_error = float(np.linalg.norm(p[:2] - [.25, .20]))
        released = not (self.pads & touched)
        placed = self.place.update(float(data.time), goal_error <= .02 and .810 < p[2] < .820 and upright and speed < .02 and angular_speed < .1 and released and self.table in touched)
        self.success = bool(self.held and self.lifted and placed)
        self.evidence = dict(role='privileged_read_only_verifier', bilateral_grasp_observed=bool(self.held),
                            sustained_lift_observed=bool(self.lifted), stable_released_placement=bool(placed),
                            success=self.success, object_position_m=p.tolist(), goal_error_m=goal_error,
                            maximum_center_height_m=self.max_height, upright=bool(upright),
                            released=bool(released), table_supported=self.table in touched,
                            geometry='primitive frictional finger pads and 44 mm solid mug cylinder; hardware calibration pending')
        return self.success
