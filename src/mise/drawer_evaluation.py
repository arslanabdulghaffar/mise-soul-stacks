"""Independent contact provenance and released drawer travel verification."""
import numpy as np
from .evaluation import StablePredicate


class DrawerContactEvaluator:
    def __init__(self, model):
        import mujoco
        self.mj, self.model = mujoco, model
        self.address = model.joint('drawer_joint').qposadr[0]
        self.handle = model.geom('drawer_handle').id
        self.pads = {model.geom(f'arm_a_{pad}_pad').id for pad in ('fixed', 'moving')}
        self.stable = StablePredicate(.5, .1)
        self.held = self.pulled = self.success = False
        self.evidence = {}

    def update(self, data):
        touched = set()
        for index, contact in enumerate(data.contact):
            if self.handle not in (contact.geom1, contact.geom2):
                continue
            force = np.zeros(6)
            self.mj.mj_contactForce(self.model, data, index, force)
            if force[0] > .005:
                touched.add(contact.geom2 if contact.geom1 == self.handle else contact.geom1)
        bilateral = self.pads <= touched
        travel = float(data.qpos[self.address])
        self.held |= bilateral
        self.pulled |= bilateral and travel > .05
        released = not (self.pads & touched)
        stable = self.stable.update(float(data.time), travel >= .144 and released)
        self.success = bool(self.held and self.pulled and stable)
        self.evidence = dict(role='privileged_read_only_verifier', success=self.success,
                             bilateral_grasp_observed=bool(self.held),
                             opening_with_bilateral_contact=bool(self.pulled),
                             stable_released_opening=bool(stable), drawer_travel_m=travel,
                             released=bool(released), drawer_actuator_present=False,
                             geometry='finite finger pads, rigid cylindrical knob and passive sliding tray')
        return self.success
