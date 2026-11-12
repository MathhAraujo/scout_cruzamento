"""Verifica unidades, descontinuidades e cardinalidade do cruzamento real."""
import unittest

import numpy as np
import pandas as pd

from pipeline import load_points, segment_motion


def trajectory(frames, fps=25):
    return pd.DataFrame({"frame":frames, "x_m":np.asarray(frames)*2/fps, "y_m":0.,
                         "team":"left", "role":"player", "coordinate_qc":True})


class KinematicsTest(unittest.TestCase):
    def test_constant_motion_units(self):
        points, segments = segment_motion(trajectory(np.arange(100)), 25)
        self.assertEqual(segments, 1)
        np.testing.assert_allclose(points.speed_m_s, 2., atol=1e-10)
        np.testing.assert_allclose(points.tangential_accel_m_s2, 0., atol=1e-10)

    def test_gap_is_not_bridged(self):
        data = trajectory(np.r_[np.arange(50), np.arange(70,120)])
        points, segments = segment_motion(data, 25)
        self.assertEqual(segments, 2)
        self.assertFalse(points.frame.between(50,69).any())
        np.testing.assert_allclose(points.speed_m_s, 2., atol=1e-10)

    def test_frame_join_rejects_unmatched_annotation(self):
        obj = {"info":{"version":"1.3", "frame_rate":25},
               "images":[{"image_id":"1", "file_name":"000001.jpg"}],
               "annotations":[{"image_id":"2", "track_id":1, "attributes":{"role":"player"},
                               "bbox_pitch":{"x_bottom_middle":0., "y_bottom_middle":0.}}]}
        with self.assertRaises(AssertionError):
            load_points(obj)


if __name__ == "__main__":
    unittest.main()
