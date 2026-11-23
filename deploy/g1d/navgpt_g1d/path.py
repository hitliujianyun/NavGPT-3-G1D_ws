"""World-frame reference paths built from successive VLA predictions.

Each prediction is anchored at the pose the robot had when its image was
captured, so it is valid in the odometry frame however late it arrives. A new
prediction is spliced onto the path being tracked:

1. both paths are cut at the robot's projection, dropping what is behind it;
2. over ``blend_m`` of arc length the shape moves from the old path to the new
   one with a smoothstep weight, so curvature changes gradually;
3. the remaining offset between the robot and the blended start decays over
   the same distance, so the path always starts exactly at the robot.

Beyond the blend the new prediction is used unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Pose, body_to_world, wrap_angle


@dataclass(frozen=True)
class RefPath:
    xy: np.ndarray            # (N, 2) world-frame points, N >= 1
    s: np.ndarray             # (N,) cumulative arc length, s[0] == 0
    end_yaw: float            # heading the prediction ends with
    stamp: float = 0.0        # capture time of the prediction it came from

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def point(self, s: float | np.ndarray) -> np.ndarray:
        s = np.clip(s, 0.0, self.length)
        return np.stack([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])], axis=-1)

    def heading(self, s: float) -> float:
        """Tangent direction at arc length s (end_yaw on degenerate paths)."""
        if self.length < 1e-6:
            return self.end_yaw
        a = self.point(max(0.0, min(s, self.length) - 0.05))
        b = self.point(min(self.length, max(s, 0.0) + 0.05))
        d = b - a
        return math.atan2(d[1], d[0]) if np.hypot(*d) > 1e-6 else self.end_yaw

    def project(self, xy) -> float:
        """Arc length of the closest point on the path to xy."""
        if len(self.xy) == 1:
            return 0.0
        p = np.asarray(xy, dtype=float)
        a, b = self.xy[:-1], self.xy[1:]
        ab = b - a
        seg2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * ab).sum(axis=1) / seg2, 0.0, 1.0)
        d2 = ((a + t[:, None] * ab - p) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        return float(self.s[i] + t[i] * math.sqrt(seg2[i]))

    def remainder(self, s0: float, ds: float) -> "RefPath":
        """The part of the path from arc length s0 on, resampled every ds."""
        s0 = min(max(s0, 0.0), self.length)
        n = max(1, int(math.ceil((self.length - s0) / ds)) + 1)
        grid = np.linspace(s0, self.length, n) if n > 1 else np.array([s0])
        return _from_points(self.point(grid), self.end_yaw, self.stamp)


def _from_points(xy: np.ndarray, end_yaw: float, stamp: float) -> RefPath:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    seg = np.hypot(*np.diff(xy, axis=0).T) if len(xy) > 1 else np.zeros(0)
    return RefPath(xy, np.concatenate([[0.0], np.cumsum(seg)]), wrap_angle(end_yaw), stamp)


def resample(xy, end_yaw: float, ds: float, stamp: float = 0.0) -> RefPath:
    """Drop repeated points and resample a polyline every ds metres."""
    pts = [np.asarray(xy[0], dtype=float)]
    for p in np.asarray(xy, dtype=float)[1:]:
        if np.hypot(*(p - pts[-1])) > 1e-3:
            pts.append(p)
    raw = _from_points(np.array(pts), end_yaw, stamp)
    return raw.remainder(0.0, ds) if raw.length > 0 else raw


def path_from_waypoints(capture_pose: Pose, waypoints, ds: float = 0.05, stamp: float = 0.0) -> RefPath:
    """Anchor eight cumulative robot-frame waypoints at the capture-time pose."""
    pts = [capture_pose[:2]]
    end_yaw = capture_pose[2]
    for wp in waypoints:
        x, y, yaw = body_to_world(capture_pose, float(wp[0]), float(wp[1]), float(wp[2]))
        pts.append((x, y))
        end_yaw = yaw
    return resample(pts, end_yaw, ds, stamp)


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _segment_headings(path: RefPath) -> tuple[np.ndarray, np.ndarray]:
    """Unwrapped heading of each segment and the arc length at its midpoint."""
    d = np.diff(path.xy, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0])), 0.5 * (path.s[:-1] + path.s[1:])


def splice(old: RefPath | None, new: RefPath, robot_xy, blend_m: float = 0.5, ds: float = 0.05) -> RefPath:
    """Path starting at the robot, easing from ``old`` into ``new``.

    Headings, not positions, are blended: over blend_m the tangent turns from
    the old path's to the new one's with a smoothstep weight, and the path is
    integrated from the robot, so it never swerves past either. The endpoint
    error left by the latency (the robot is no longer where the prediction
    started) is spread along the whole path, so the path still ends where the
    VLA put it.
    """
    robot = np.asarray(robot_xy, dtype=float)
    new_r = new.remainder(new.project(robot), ds)
    if new_r.length < ds:
        return _from_points(robot[None, :], new.end_yaw, new.stamp)

    heading, mid = _segment_headings(new_r)
    if old is not None and blend_m > 0:
        old_r = old.remainder(old.project(robot), ds)
        if old_r.length > ds:
            old_heading, old_mid = _segment_headings(old_r)
            h_old = np.interp(np.minimum(mid, old_mid[-1]), old_mid, old_heading)
            h_old = h_old + 2 * np.pi * np.round((heading[0] - h_old[0]) / (2 * np.pi))
            a = _smoothstep(mid / min(blend_m, old_r.length))
            heading = h_old + a * (heading - h_old)

    seg = np.diff(new_r.s)
    pts = robot + np.vstack([[0.0, 0.0], np.cumsum(np.stack([seg * np.cos(heading), seg * np.sin(heading)], 1), 0)])
    pts = pts + _smoothstep(new_r.s / new_r.length)[:, None] * (new_r.xy[-1] - pts[-1])
    return _from_points(pts, new.end_yaw, new.stamp)


def deviation(a: RefPath, b: RefPath, robot_xy, upto_m: float = 1.0, ds: float = 0.05) -> float:
    """Largest distance between two paths over their first upto_m after the robot."""
    ra, rb = a.remainder(a.project(robot_xy), ds), b.remainder(b.project(robot_xy), ds)
    span = min(upto_m, ra.length, rb.length)
    if span <= 0:
        return float(np.hypot(*(ra.xy[0] - rb.xy[0])))
    grid = np.arange(0.0, span + 0.5 * ds, ds)
    return float(np.max(np.hypot(*(ra.point(grid) - rb.point(grid)).T)))
