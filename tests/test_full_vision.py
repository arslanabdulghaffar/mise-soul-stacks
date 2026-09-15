"""Regression coverage for utensil measurements at the table search boundary."""
import unittest
import numpy as np
from mise.full_vision import Bounds, locate_color


class FullVisionTests(unittest.TestCase):
    def test_search_region_does_not_clip_a_utensil_whose_center_is_inside(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[40:70, 46:52, 1] = 200
        bounds = Bounds(-.2, .2, -.08, .2)
        complete = locate_color(image, 'fork', object_top=.8, measure_axis=True)
        measured = locate_color(image, 'fork', object_top=.8, bounds=bounds,
                                measure_axis=True, preserve_components=True)
        clipped = locate_color(image, 'fork', object_top=.8, bounds=bounds)
        self.assertEqual(measured.xy, complete.xy)
        self.assertEqual(measured.pixels, 180)
        self.assertEqual(measured.axis_error_degrees, 0)
        self.assertEqual(measured.heading_radians, 0)
        self.assertGreater(clipped.xy[1], measured.xy[1] + .02)

    def test_larger_component_outside_search_region_does_not_hide_valid_utensil(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[40:70, 46:52, 1] = 200
        image[5:25, 5:25, 1] = 200
        measured = locate_color(image, 'fork', object_top=.8, bounds=Bounds(-.2,.2,-.08,.2),
                                preserve_components=True)
        self.assertEqual(measured.pixels, 180)
        self.assertEqual(measured.centroid, (48.5, 54.5))
