from __future__ import annotations

import unittest

import numpy as np

from mise.recovery_labels import ForkLabeler, MuJoCoForkRunner, MuJoCoSnapshot
from mise.sim import BimanualTableEnv
from mise.types import RecoverabilityVerdict


class RecoveryLabelTests(unittest.TestCase):
    def test_viable_current_arm_continues(self) -> None:
        labeler = ForkLabeler(forks_per_arm=4, success_threshold=0.60)
        decision, stats = labeler.label("A", lambda arm, index: arm == "A" or index == 0)
        self.assertEqual(decision.verdict, RecoverabilityVerdict.CONTINUE)
        self.assertEqual(stats.current_success_rate, 1.0)

    def test_alternate_arm_triggers_replan(self) -> None:
        labeler = ForkLabeler(forks_per_arm=4, success_threshold=0.60)
        decision, stats = labeler.label("A", lambda arm, index: arm == "B" and index < 3)
        self.assertEqual(decision.verdict, RecoverabilityVerdict.REPLAN)
        self.assertEqual(stats.alternate_success_rate, 0.75)

    def test_no_viable_arm_stops_without_tested_corrective_action(self) -> None:
        labeler = ForkLabeler(forks_per_arm=4, success_threshold=0.60)
        decision, _ = labeler.label("B", lambda arm, index: index == 0)
        self.assertEqual(decision.verdict, RecoverabilityVerdict.STOP)

    def test_mujoco_fork_does_not_mutate_observed_state(self) -> None:
        env = BimanualTableEnv(seed=4)
        try:
            snapshot = MuJoCoSnapshot.capture(env.data)
            runner = MuJoCoForkRunner(env._mj, env.model, horizon=3)

            def controller(arm, model, data, step):
                data.ctrl[model.actuator("drawer_slide").id] = 0.18 if arm == "A" else 0.0

            success = lambda model, data: data.time > snapshot.time
            self.assertTrue(runner.rollout_from(snapshot, "A", 0, controller, success))
            np.testing.assert_allclose(env.data.qpos, snapshot.qpos)
            self.assertAlmostEqual(env.data.time, snapshot.time)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
