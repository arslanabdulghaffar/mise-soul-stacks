"""Contact integration tests: actual MuJoCo dynamics and RGB, no scripted objects."""
import unittest
import numpy as np
from mise.contact_evaluation import ContactSkillEvaluator
from mise.contact_vision import locate_mug
from mise.manipulation import CONTACT_COMMAND, ManipulationController, create_contact_env
from mise.planner import RuleBasedPlanner


class ManipulationTests(unittest.TestCase):
    def test_missing_visual_object_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not sufficiently visible'):
            locate_mug(np.zeros((256, 256, 3), dtype=np.uint8))

    def test_camera_guided_physical_placement_across_starts(self):
        observed_starts = []
        for seed, preset in ((1001, 'nominal'), (1002, 'nominal'), (1003, 'displaced'), (1004, 'low_friction')):
            with self.subTest(seed=seed, preset=preset):
                env = create_contact_env(seed)
                try:
                    address = env.model.joint('mug_free').qposadr[0]
                    if preset == 'displaced':
                        env.data.qpos[address] += .02  # Initial condition, before execution.
                        env._mj.mj_forward(env.model, env.data)
                    elif preset == 'low_friction':
                        env.model.geom_friction[env.model.geom_bodyid == env.model.body('mug').id] *= .7
                    initial = env.data.qpos[address:address + 3].copy()
                    self.assertGreater(np.linalg.norm(initial[:2] - [.25, .2]), .15)
                    self.assertEqual(env.model.nu, 12)  # Robot actuators only, no drawer/object servo.
                    self.assertEqual(env.model.neq, 0)  # No grasp weld or attachment constraint.
                    controller = ManipulationController(env, RuleBasedPlanner().plan(CONTACT_COMMAND))
                    verifier = ContactSkillEvaluator(env.model)
                    for _ in range(750):
                        before = env.data.qpos.copy()
                        action = controller.advance()
                        np.testing.assert_array_equal(env.data.qpos, before)
                        if controller.done or controller.failed_reason:
                            break
                        env.step(action, observe=False)
                        before = env.data.qpos.copy()
                        verifier.update(env.data)
                        np.testing.assert_array_equal(env.data.qpos, before)
                    self.assertIsNone(controller.failed_reason)
                    self.assertTrue(controller.done)
                    self.assertTrue(verifier.success, verifier.evidence)
                    self.assertTrue(verifier.evidence['bilateral_grasp_observed'])
                    self.assertTrue(verifier.evidence['sustained_lift_observed'])
                    self.assertLess(verifier.evidence['goal_error_m'], .02)
                    np.testing.assert_array_equal(env.data.xfrc_applied, 0)
                    np.testing.assert_array_equal(env.data.qfrc_applied, 0)
                    observed_starts.append(controller.evidence['initial_detection']['xy'])
                    # A goal-only scene cannot manufacture lift/grasp provenance.
                    fresh_verifier = ContactSkillEvaluator(env.model)
                    self.assertFalse(fresh_verifier.update(env.data))
                finally:
                    env.close()
        self.assertGreater(np.linalg.norm(np.asarray(observed_starts[2]) - observed_starts[0]), .01)
