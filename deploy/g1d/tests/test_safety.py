import math
import unittest

import numpy as np

from navgpt_g1d.dds import Scan
from navgpt_g1d.safety import SafetyParams, ScanSafety


def scan_with_point(x, y, now=0.0):
    return Scan(now, np.array([math.atan2(y, x)], np.float32), np.array([math.hypot(x, y)], np.float32))


class SafetyTest(unittest.TestCase):
    p = SafetyParams(front_m=0.3, stop_dist_m=0.35, slow_dist_m=1.2, half_width_m=0.27, margin_m=0.1)

    def check(self, scan, v=0.3, w=0.0, now=0.0):
        return ScanSafety(self.p, v_max=0.3).check(scan, now, v, w)

    def test_clear_path(self):
        self.assertFalse(self.check(scan_with_point(5.0, 0.0)).stop)
        self.assertIsNone(self.check(scan_with_point(0.5, 1.0)).v_cap, "beside the corridor")

    def test_stop_zone_grows_with_speed(self):
        x = 0.3 + 0.40                            # 0.40 m beyond the bumper
        self.assertFalse(self.check(scan_with_point(x, 0.0), v=0.0).stop)
        self.assertTrue(self.check(scan_with_point(x, 0.0), v=0.3).stop, "0.35 + 0.3 x 0.3 = 0.44 m")

    def test_slow_zone_scales_speed(self):
        near = self.check(scan_with_point(0.3 + 0.6, 0.0), v=0.1).v_cap
        far = self.check(scan_with_point(0.3 + 1.0, 0.0), v=0.1).v_cap
        self.assertLess(near, far)
        self.assertLess(far, 0.3)

    def test_rotation_circle(self):
        side = scan_with_point(0.0, 0.38)
        self.assertTrue(self.check(side, v=0.0, w=0.4).stop)
        self.assertFalse(self.check(side, v=0.3, w=0.0).stop, "outside the forward corridor")

    def test_stale_scan_stops(self):
        self.assertTrue(self.check(scan_with_point(5.0, 0.0, now=0.0), now=1.0).stop)
        self.assertTrue(self.check(None).stop)

    def test_self_returns_ignored(self):
        self.assertFalse(self.check(scan_with_point(0.05, 0.0)).stop)


if __name__ == "__main__":
    unittest.main()
