"""Frozen stress fixtures must change physics and preserve visual localization."""
from pathlib import Path
from runpy import run_path
import tempfile
import unittest

import numpy as np

from mise.full_task import create_full_env
from mise.full_vision import locate_plate
from mise.sim import ROOT
from scripts.evaluate_robustness import variant


class RobustnessTests(unittest.TestCase):
    def test_shape_variant_is_reproducible_and_used_by_simulator(self):
        base = run_path(str(ROOT / 'scripts/build_full_scene.py'))['build_full_scene']()
        with tempfile.TemporaryDirectory() as directory:
            one, two = (Path(directory) / name for name in ('one.xml', 'two.xml'))
            self.assertEqual(variant(base, one, 1001, 'shape'), variant(base, two, 1001, 'shape'))
            self.assertEqual(one.read_bytes(), two.read_bytes())
            env = create_full_env(1001, scene_path=one)
            try:
                for name in ('mug', 'spoon_bowl'):
                    self.assertEqual(env.model.geom(name).type[0], env._mj.mjtGeom.mjGEOM_BOX)
                self.assertEqual(env.model.nu, 12)
                self.assertEqual(env.model.neq, 0)
            finally:
                env.close()

    def test_gray_background_highlights_do_not_become_plate_target(self):
        base = run_path(str(ROOT / 'scripts/build_full_scene.py'))['build_full_scene']()
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'gray.xml'
            variant(base, scene, 1001, 'background')
            env = create_full_env(1001, scene_path=scene)
            try:
                detection = locate_plate(env.render('top'))
                # Privileged state is read only by this test to score the RGB estimate.
                actual = env.data.xpos[env.model.body('plate').id, :2]
                self.assertLess(np.linalg.norm(np.asarray(detection.xy) - actual), .01)
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
