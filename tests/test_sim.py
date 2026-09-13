from __future__ import annotations

import unittest

import numpy as np

from mise.sim import BimanualTableEnv, build_scene


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_scene()

    def test_scene_exposes_three_cameras_and_twelve_joints(self) -> None:
        env = BimanualTableEnv(seed=7)
        try:
            observation = env.observe()
            self.assertEqual(observation.top.shape, (256, 256, 3))
            self.assertEqual(observation.wrist_a.shape, (256, 256, 3))
            self.assertEqual(observation.wrist_b.shape, (256, 256, 3))
            self.assertEqual(observation.joint_positions.shape, (12,))
            env.step(np.zeros(12), frames=3)
        finally:
            env.close()

    def test_drawer_actuator_opens_the_drawer(self) -> None:
        env = BimanualTableEnv(seed=7)
        try:
            env.set_drawer(0.18)
            for _ in range(30):
                env.step(np.zeros(12), observe=False)
            drawer_qpos = env.data.qpos[env.model.joint("drawer_joint").qposadr[0]]
            self.assertGreater(drawer_qpos, 0.15)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
