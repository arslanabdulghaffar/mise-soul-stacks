"""Verify presentation does not perturb physics or calibrated controller pixels."""
import unittest

import numpy as np

from mise.presentation import CAMERAS, PROFILES, PresentationRenderer, capture_settings
from mise.sim import BimanualTableEnv, build_scene


class PresentationTests(unittest.TestCase):
    def test_display_cameras_preserve_controller_observation_and_state(self):
        env = BimanualTableEnv(seed=1001, scene_path=build_scene())
        try:
            before = env.observe()
            qpos, qvel, ctrl = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
            sim_time = env.data.time
            for profile in PROFILES:
                settings = capture_settings(profile)
                size = settings["width"]
                self.assertEqual(len(set(settings["capture_every"].values())), 1)
                renderer = PresentationRenderer(env, size, camera_dimensions=settings["camera_dimensions"])
                try:
                    for camera in CAMERAS:
                        frame = renderer.render(camera)
                        dimensions = settings["camera_dimensions"][camera]
                        self.assertEqual(frame.shape, (dimensions["height"], dimensions["width"], 3))
                        self.assertGreater(frame.std(), 5, "Camera must contain a visible scene")
                finally:
                    renderer.close()
                after = env.observe()
                for name in ("top", "wrist_a", "wrist_b", "joint_positions"):
                    np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
                for expected, actual in ((qpos, env.data.qpos), (qvel, env.data.qvel), (ctrl, env.data.ctrl)):
                    np.testing.assert_array_equal(expected, actual)
                self.assertEqual(env.data.time, sim_time)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
