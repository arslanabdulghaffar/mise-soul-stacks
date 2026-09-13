import unittest
import numpy as np
from mise.drawer_skill import locate_handle, DRAWER_COMMAND
from mise.planner import RuleBasedPlanner
from mise.skills import create_skill_env, create_skill_controller, create_skill_evaluator
from mise.drawer_evaluation import DrawerContactEvaluator


class DrawerSkillTests(unittest.TestCase):
    def test_occluded_handle_does_not_generate_targets(self):
        with self.assertRaisesRegex(ValueError, 'not sufficiently visible'):
            locate_handle(np.zeros((256, 256, 3), dtype=np.uint8))

    def test_camera_guided_passive_drawer_has_grasp_and_pull_provenance(self):
        plan = RuleBasedPlanner().plan(DRAWER_COMMAND)
        env = create_skill_env(plan, 1001)
        try:
            self.assertEqual(env.model.nu, 12)
            self.assertEqual(env.model.neq, 0)
            controller, evaluator = create_skill_controller(env, plan), create_skill_evaluator(env, plan)
            for _ in range(900):
                before = env.data.qpos.copy()
                action = controller.advance()
                np.testing.assert_array_equal(env.data.qpos, before)
                if controller.done or controller.failed_reason:
                    break
                env.step(action, observe=False)
                before = env.data.qpos.copy()
                evaluator.update(env.data)
                np.testing.assert_array_equal(env.data.qpos, before)
            self.assertIsNone(controller.failed_reason)
            self.assertTrue(controller.done)
            self.assertTrue(evaluator.success, evaluator.evidence)
            self.assertTrue(evaluator.pulled)
            self.assertGreaterEqual(controller.evidence['visual_drawer_travel_m'], .144)
            np.testing.assert_array_equal(env.data.xfrc_applied, 0)
            np.testing.assert_array_equal(env.data.qfrc_applied, 0)
            fresh = DrawerContactEvaluator(env.model)
            self.assertFalse(fresh.update(env.data), 'An already open drawer is not grasp provenance.')
        finally:
            env.close()
