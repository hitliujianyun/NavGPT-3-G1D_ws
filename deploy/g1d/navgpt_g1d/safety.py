"""Laser-scan protection for the G1-D base, evaluated every control tick.

Two zones in front of the chassis, both as wide as the chassis plus a margin:

* slow zone: an obstacle at distance d caps the speed at
  v_max * (d - stop_dist) / (slow_dist - stop_dist), so the robot eases toward
  people and furniture instead of braking hard at the last moment;
* stop zone: stop_dist plus the distance covered in react_s at the current
  speed; anything inside raises a protective stop.

When turning in place, anything inside the rotation circle plus a margin stops
the robot. The base never drives backwards (the tracker only moves forward), so
only the front is checked while driving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SafetyParams:
    enabled: bool = True
    half_width_m: float = 0.27       # chassis 525 mm wide
    margin_m: float = 0.10
    front_m: float = 0.30            # lidar centre to the front bumper
    stop_dist_m: float = 0.35        # beyond the bumper
    slow_dist_m: float = 1.20        # beyond the bumper
    react_s: float = 0.3
    rotate_radius_m: float = 0.34    # 666 mm rotation diameter
    rotate_margin_m: float = 0.08
    min_range_m: float = 0.12        # closer returns are the robot itself
    lidar_x_m: float = 0.0           # lidar position in the base frame
    lidar_y_m: float = 0.0
    lidar_yaw_rad: float = 0.0
    scan_timeout_s: float = 0.5
    resume_clear_s: float = 1.0      # zone clear this long before motion may resume


@dataclass(frozen=True)
class SafetyVerdict:
    stop: bool
    v_cap: Optional[float]
    reason: str = ""
    nearest_m: float = math.inf


class ScanSafety:
    def __init__(self, params: SafetyParams, v_max: float):
        self.p, self.v_max = params, v_max

    def points(self, scan) -> np.ndarray:
        p = self.p
        keep = scan.ranges > p.min_range_m
        a = scan.angles[keep] + p.lidar_yaw_rad
        r = scan.ranges[keep]
        return np.stack([p.lidar_x_m + r * np.cos(a), p.lidar_y_m + r * np.sin(a)], axis=1)

    def check(self, scan, now: float, v: float, w: float) -> SafetyVerdict:
        p = self.p
        if not p.enabled:
            return SafetyVerdict(False, None)
        if scan is None or now - scan.stamp > p.scan_timeout_s:
            return SafetyVerdict(True, 0.0, "laser scan stale")
        pts = self.points(scan)
        if len(pts) == 0:
            return SafetyVerdict(False, None)

        if abs(v) < 0.02 and abs(w) > 0.05:
            r = np.hypot(pts[:, 0], pts[:, 1])
            if np.any(r < p.rotate_radius_m + p.rotate_margin_m):
                return SafetyVerdict(True, 0.0, "obstacle inside rotation circle", float(r.min()))

        corridor = (pts[:, 0] > 0.0) & (np.abs(pts[:, 1]) < p.half_width_m + p.margin_m)
        if not np.any(corridor):
            return SafetyVerdict(False, None)
        ahead = float(pts[corridor, 0].min()) - p.front_m
        stop_dist = p.stop_dist_m + max(v, 0.0) * p.react_s
        if ahead < stop_dist:
            return SafetyVerdict(True, 0.0, "obstacle {:.2f} m ahead".format(ahead), ahead)
        if ahead < p.slow_dist_m:
            frac = (ahead - p.stop_dist_m) / (p.slow_dist_m - p.stop_dist_m)
            return SafetyVerdict(False, max(0.05, self.v_max * frac), "slowing", ahead)
        return SafetyVerdict(False, None, "", ahead)

    def clear_for_resume(self, scan, now: float) -> bool:
        verdict = self.check(scan, now, 0.0, 0.0)
        return not verdict.stop and (verdict.nearest_m - self.p.stop_dist_m > 0.05)
