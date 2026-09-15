"""Controller safety and chunk timing without a trained-model dependency."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mise.learned_controller import LearnedContactController
from mise.planner import RuleBasedPlanner


class LearnedControllerTests(unittest.TestCase):
    def controller(self, chunks, **settings):
        joints = np.zeros(12)
        observation = SimpleNamespace(joint_positions=joints, top=np.zeros((32, 32, 3)))
        env = SimpleNamespace(data=SimpleNamespace(time=0.),
                              model=SimpleNamespace(actuator_ctrlrange=np.tile([-.5, .5], (12, 1))),
                              _arm_ctrl=np.arange(12), joint_positions=lambda: joints.copy(),
                              observe=lambda: observation)
        ik = SimpleNamespace(scratch=SimpleNamespace(qpos=np.zeros(5), site_xpos=np.zeros((1, 3))),
                             qpos=np.arange(5), site=0, mj=Mock())
        policy = Mock()
        policy.predict_chunk.side_effect = chunks
        with patch('mise.learned_controller.ArmKinematics', return_value=ik):
            controller = LearnedContactController(
                env, RuleBasedPlanner().plan('Place the mug in the upper-right with arm B.'), policy, **settings)
        return controller, policy

    def test_replanning_keeps_overlapping_predictions_aligned(self):
        first = np.full((15, 12), .02)
        second = np.full((15, 12), -.02)
        # With averaging enabled the new action lies between both predictions;
        # without it the new chunk takes over immediately at tick 5.
        for ensemble in (False, True):
            controller, policy = self.controller([first, second], execution_steps=5, temporal_ensemble=ensemble)
            with patch('mise.learned_controller.locate_mug', side_effect=ValueError('not visible')):
                for _ in range(5):
                    np.testing.assert_allclose(controller.advance(), .02)
                self.assertEqual(policy.predict_chunk.call_count, 1)
                action = controller.advance()
            self.assertEqual(policy.predict_chunk.call_count, 2)
            self.assertIsNone(controller.failed_reason)
            if ensemble:
                self.assertTrue(np.all((action > -.02) & (action < .02)))
            else:
                np.testing.assert_allclose(action, -.02)

    def test_invalid_predictions_hold_previous_action_and_fail(self):
        for chunk in (np.zeros((4, 12)), np.full((15, 12), np.nan), np.zeros((15, 11))):
            controller, _ = self.controller([chunk], execution_steps=5)
            with patch('mise.learned_controller.locate_mug', side_effect=ValueError('not visible')):
                np.testing.assert_array_equal(controller.advance(), np.zeros(12))
                np.testing.assert_array_equal(controller.advance(), np.zeros(12))
            self.assertIn('invalid joint-target chunk', controller.failed_reason)

    def test_predictions_cannot_bypass_action_delta_guard(self):
        controller, _ = self.controller([np.full((15, 12), 100.)])
        with patch('mise.learned_controller.locate_mug', side_effect=ValueError('not visible')):
            np.testing.assert_allclose(controller.advance(), .08)
            for _ in range(10):
                action = controller.advance()
        np.testing.assert_allclose(action, .5)


if __name__ == '__main__':
    unittest.main()
