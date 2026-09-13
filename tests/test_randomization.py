from __future__ import annotations

import unittest

import numpy as np

from mise.sim import BimanualTableEnv


class RandomizationTests(unittest.TestCase):
    def test_reset_is_reproducible_for_a_fixed_seed(self) -> None:
        env = BimanualTableEnv(seed=0)
        try:
            env.reset(seed=42)
            first_state = env.data.qpos.copy()
            first_metadata = env.last_randomization
            env.reset(seed=42)
            np.testing.assert_allclose(env.data.qpos, first_state)
            self.assertEqual(env.last_randomization, first_metadata)
        finally:
            env.close()

    def test_drawer_starts_open_at_probability_one(self) -> None:
        env = BimanualTableEnv(seed=0)
        try:
            env.randomizer.config = env.randomizer.config.__class__(drawer_open_probability=1.0)
            env.reset(seed=3)
            self.assertTrue(env.last_randomization and env.last_randomization.drawer_open)
            drawer_qpos = env.data.qpos[env.model.joint("drawer_joint").qposadr[0]]
            self.assertAlmostEqual(drawer_qpos, 0.18)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

