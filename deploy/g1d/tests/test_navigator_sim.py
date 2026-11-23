"""Kinematic simulation of the Navigator alone (no threads), against stop-and-go.

A fake VLA sees the robot's true pose at capture time and predicts eight
waypoints 0.15 m apart along an L-shaped route, with noise. Replies arrive
0.4-1.2 s later, with occasional 2.5 s Wi-Fi stalls. Odometry is sampled at
10 Hz, control runs at 20 Hz.
"""

import math
import random
import unittest

import numpy as np

from navgpt_g1d.geometry import integrate_unicycle, world_to_body, wrap_angle
from navgpt_g1d.navigator import Navigator, suggests_arrival
from navgpt_g1d.path import resample
from navgpt_g1d.pose_buffer import PoseBuffer

ROUTE = resample([(0.0, 0.0), (3.0, 0.0), (3.0, 2.5)], math.pi / 2, 0.02)
DT = 0.05


def fake_vla(pose, rng):
    s0 = ROUTE.project(pose[:2])
    if ROUTE.length - s0 < 0.3:
        return [[0.02, 0.0, 0.0]] * 8
    wps = []
    for k in range(1, 9):
        s = s0 + 0.15 * k
        wx, wy = ROUTE.point(s)
        bx, by = world_to_body(pose, wx, wy)
        theta = wrap_angle(ROUTE.heading(s) - pose[2])
        wps.append([bx + rng.gauss(0, 0.02), by + rng.gauss(0, 0.04), theta + rng.gauss(0, 0.05)])
    return wps


def latency(rng):
    return 2.5 if rng.random() < 0.08 else rng.uniform(0.4, 1.2)


def run_overlapped(seed=0, t_max=90.0):
    rng = random.Random(seed)
    nav, odom = Navigator(), PoseBuffer()
    pose, t, pending = (0.0, 0.0, 0.0), 0.0, None
    log = []
    while t < t_max and not nav.finished:
        if round(t / DT) % 2 == 0:
            odom.add(t, pose)
        if pending is None:
            req = nav.next_request(t, odom.at(t))
            pending = (req, fake_vla(pose, rng), t + latency(rng))
        elif t >= pending[2]:
            nav.on_reply(pending[0], pending[1], odom.predict(t), t)
            pending = None
        v, w = nav.control(odom.predict(t), t, DT)
        odom.set_command(v, w)
        pose = integrate_unicycle(pose, v, w, DT)
        log.append((t, v, w, pose, t - nav.last_reply_time))
        t += DT
    return nav, log


def arc_to(x, y, v_max=0.3, w_max=0.5):
    """Constant (v, w, duration) reaching (x, y) along an arc, turning in place if far off-axis."""
    bearing = math.atan2(y, x)
    if abs(bearing) > math.radians(50):
        return 0.0, math.copysign(w_max, bearing), abs(bearing) / w_max
    dist = math.hypot(x, y)
    angle = 2 * bearing
    length = dist if abs(y) < 1e-6 else abs(dist * dist / (2 * y) * angle)
    duration = max(0.5, length / v_max, abs(angle) / w_max)
    return length / duration, angle / duration, duration


def run_stop_and_go(seed=0, t_max=200.0):
    """Capture, wait for the reply, drive to waypoint 1, stop, repeat."""
    rng = random.Random(seed)
    pose, t, streak = (0.0, 0.0, 0.0), 0.0, 0
    while t < t_max:
        wps = fake_vla(pose, rng)
        t += latency(rng)
        streak = streak + 1 if suggests_arrival(wps) else 0
        if streak >= 2:
            break
        v, w, duration = arc_to(*wps[1][:2])
        pose = integrate_unicycle(pose, v, w, duration)
        t += duration
    return t, pose


class NavigatorSimTest(unittest.TestCase):
    def test_overlap_is_smooth_and_faster(self):
        nav, log = run_overlapped()
        self.assertTrue(nav.finished, "reached the end and stopped")
        t_end, _, _, end_pose, _ = log[-1]
        self.assertLess(math.hypot(end_pose[0] - 3.0, end_pose[1] - 2.5), 0.35)

        v = np.array([r[1] for r in log])
        w = np.array([r[2] for r in log])
        age = np.array([r[4] for r in log])
        p = nav.p.tracker
        self.assertLessEqual(np.max(np.abs(np.diff(v))) / DT, max(p.a_max, p.a_dec) + 1e-9)
        self.assertLessEqual(np.max(np.abs(np.diff(w))) / DT, p.alpha_max + 1e-9)

        moving = np.nonzero(v > 0.05)[0]
        span = slice(moving[0], moving[-1])
        linked = age[span] < nav.p.starve_s
        stalls = float(np.mean(v[span][linked] < 0.05))
        self.assertLess(stalls, 0.02, "no stop-and-go while predictions keep arriving")

        t_sg, _ = run_stop_and_go()
        self.assertLess(t_end, 0.75 * t_sg)

    def test_several_seeds_finish(self):
        for seed in range(1, 6):
            nav, _ = run_overlapped(seed)
            self.assertTrue(nav.finished, "seed {}".format(seed))


if __name__ == "__main__":
    unittest.main()
