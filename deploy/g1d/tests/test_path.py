import math
import unittest

import numpy as np

from navgpt_g1d.navigator import Navigator
from navgpt_g1d.path import path_from_waypoints, resample, splice
from navgpt_g1d.pose_buffer import PoseBuffer


def max_turn_per_segment(path) -> float:
    d = np.diff(path.xy, axis=0)
    d = d[np.hypot(*d.T) > 1e-6]
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.max(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi))) if len(ang) > 1 else 0.0


class PathTest(unittest.TestCase):
    def test_anchor_at_capture_pose(self):
        path = path_from_waypoints((1.0, 2.0, math.pi / 2), [[0.25, 0, 0], [0.5, 0, 0.1]])
        np.testing.assert_allclose(path.xy[0], [1.0, 2.0], atol=1e-9)
        np.testing.assert_allclose(path.xy[-1], [1.0, 2.5], atol=1e-9)
        self.assertAlmostEqual(path.end_yaw, math.pi / 2 + 0.1)

    def test_projection(self):
        path = resample([(0, 0), (1, 0), (1, 1)], 0.0, 0.05)
        self.assertAlmostEqual(path.project((0.4, 0.2)), 0.4, places=6)
        self.assertAlmostEqual(path.project((1.3, 0.5)), 1.5, places=6)

    def test_splice_starts_at_robot_and_converges_to_new(self):
        new = resample([(0, 0), (2, 0)], 0.0, 0.05)
        robot = (0.3, 0.12)                      # robot drifted 12 cm off the new path
        out = splice(None, new, robot, blend_m=0.5)
        np.testing.assert_allclose(out.xy[0], robot, atol=1e-9)
        np.testing.assert_allclose(out.xy[-1], new.xy[-1], atol=1e-9)
        y = out.xy[:, 1]
        self.assertTrue(np.all(np.diff(y) <= 1e-9) and np.all(y >= -1e-9), "rejoins without overshoot")

    def test_splice_eases_between_disagreeing_predictions(self):
        old = resample([(0, 0), (2, 0)], 0.0, 0.05)
        new = resample([(0, 0), (2 * math.cos(0.5), 2 * math.sin(0.5))], 0.5, 0.05)  # 29 deg apart
        out = splice(old, new, (0.4, 0.0), blend_m=0.5)
        np.testing.assert_allclose(out.xy[0], (0.4, 0.0), atol=1e-9)
        self.assertLess(max_turn_per_segment(out), math.radians(8),
                        "heading changes gradually instead of a 29 deg kink")
        np.testing.assert_allclose(out.xy[-1], new.xy[-1], atol=1e-9)
        start_heading = math.atan2(*(out.xy[1] - out.xy[0])[::-1])
        self.assertLess(abs(start_heading), math.radians(3), "starts along the path being tracked")

    def test_robot_past_short_prediction(self):
        new = resample([(0, 0), (0.2, 0)], 0.0, 0.05)
        out = splice(None, new, (0.5, 0.0))
        self.assertLess(out.length, 1e-6)


class NavigatorTest(unittest.TestCase):
    def test_stale_and_reordered_replies_rejected(self):
        nav = Navigator()
        r1 = nav.next_request(0.0, (0, 0, 0))
        r2 = nav.next_request(0.5, (0, 0, 0))
        wps = [[0.15 * k, 0, 0] for k in range(1, 9)]
        self.assertTrue(nav.on_reply(r2, wps, (0, 0, 0), now=1.0))
        self.assertFalse(nav.on_reply(r1, wps, (0, 0, 0), now=1.1), "older request arriving late")
        r3 = nav.next_request(1.0, (0, 0, 0))
        self.assertFalse(nav.on_reply(r3, wps, (0, 0, 0), now=10.0), "reply older than max age")

    def test_keyframe_commit_spacing(self):
        nav = Navigator()
        self.assertTrue(nav.next_request(0.0, (0, 0, 0)).commit)
        self.assertFalse(nav.next_request(0.1, (0.1, 0, 0)).commit)
        self.assertTrue(nav.next_request(0.2, (0.26, 0, 0)).commit)
        self.assertTrue(nav.next_request(0.3, (0.26, 0, math.radians(16))).commit)


class PoseBufferTest(unittest.TestCase):
    def test_interpolation_across_wrap(self):
        buf = PoseBuffer()
        buf.add(0.0, (0.0, 0.0, math.pi - 0.1))
        buf.add(1.0, (1.0, 0.0, -math.pi + 0.1))
        x, _, yaw = buf.at(0.5)
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(abs(yaw), math.pi, places=6)

    def test_dead_reckoning(self):
        buf = PoseBuffer()
        buf.add(0.0, (0.0, 0.0, 0.0))
        buf.set_command(0.3, 0.0)
        self.assertAlmostEqual(buf.predict(0.1)[0], 0.03)
        self.assertAlmostEqual(buf.predict(5.0)[0], 0.09, msg="extrapolation is capped")


if __name__ == "__main__":
    unittest.main()

