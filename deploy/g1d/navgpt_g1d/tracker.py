"""Continuous path tracking for the G1-D differential-drive base (20 Hz).

Pure pursuit on the spliced reference path, with three speed limits:
curvature (vyaw <= w_max), stopping distance on the remaining path
(v <= sqrt(2 a_dec s)), and the configured maximum; then acceleration limits on
both v and vyaw. Intermediate waypoints are never stopped at or aligned to: only
a ``final`` path ends with a stop and a heading alignment. A path that is about
to run out slows the robot down smoothly instead of halting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose, wrap_angle
from .path import RefPath


@dataclass(frozen=True)
class TrackerParams:
    v_max: float = 0.3            # m/s
    w_max: float = 0.5            # rad/s (base limit 0.6)
    a_max: float = 0.4            # m/s^2, speeding up
    a_dec: float = 0.6            # m/s^2, slowing down
    alpha_max: float = 1.2        # rad/s^2
    lookahead_min: float = 0.35   # m
    lookahead_time: float = 1.0   # s, lookahead grows with speed
    rotate_enter: float = 1.0     # rad, turn in place above this bearing...
    rotate_exit: float = 0.35     # ...until back below this one
    k_rotate: float = 1.5         # vyaw per rad of bearing while turning in place
    goal_tol: float = 0.10        # m, final position tolerance
    yaw_tol: float = 0.10         # rad, final heading tolerance
    min_path_m: float = 0.08      # shorter predictions are pure rotations


def _toward(cur: float, target: float, up: float, down: float) -> float:
    if abs(target) > abs(cur) and target * cur >= 0:
        step = up
    else:
        step = down
    return cur + max(-step, min(step, target - cur))


class PathTracker:
    def __init__(self, params: TrackerParams = TrackerParams()):
        self.p = params
        self.path: RefPath | None = None
        self.final = False
        self.v = 0.0
        self.w = 0.0
        self.rotating = False
        self.done = False
        self._at_goal = False

    def set_path(self, path: RefPath | None, final: bool = False) -> None:
        self.path, self.final, self.done = path, final, False

    def stop(self) -> None:
        self.set_path(None)

    def step(self, pose: Pose, dt: float, v_cap: float | None = None) -> tuple[float, float]:
        """One control tick; v_cap lowers the speed limit (e.g. while starved of predictions)."""
        self._at_goal = False
        v_t, w_t = self._target(pose)
        if v_cap is not None and v_t > v_cap:
            w_t *= v_cap / v_t
            v_t = v_cap
        p = self.p
        self.v = _toward(self.v, v_t, p.a_max * dt, p.a_dec * dt)
        self.w = _toward(self.w, w_t, p.alpha_max * dt, p.alpha_max * dt)
        self.done = self.final and self._at_goal and abs(self.v) < 1e-3 and abs(self.w) < 1e-3
        return self.v, self.w

    def _target(self, pose: Pose) -> tuple[float, float]:
        p, path = self.p, self.path
        if path is None:
            return 0.0, 0.0

        s_robot = path.project(pose[:2])
        remaining = path.length - s_robot

        if path.length < p.min_path_m or remaining < p.goal_tol:
            if path.length >= p.min_path_m and not self.final:
                return 0.0, 0.0
            yaw_err = wrap_angle(path.end_yaw - pose[2])
            if abs(yaw_err) < p.yaw_tol:
                self._at_goal = True
                return 0.0, 0.0
            return 0.0, max(-p.w_max, min(p.w_max, p.k_rotate * yaw_err))

        lookahead = max(p.lookahead_min, p.lookahead_time * abs(self.v))
        tx, ty = path.point(s_robot + lookahead)
        dx, dy = tx - pose[0], ty - pose[1]
        dist = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - pose[2])

        if self.rotating and abs(bearing) < p.rotate_exit:
            self.rotating = False
        elif not self.rotating and abs(bearing) > p.rotate_enter:
            self.rotating = True
        if self.rotating:
            return 0.0, max(-p.w_max, min(p.w_max, p.k_rotate * bearing))

        curvature = 2.0 * math.sin(bearing) / max(dist, 1e-3)
        v_curve = p.w_max / abs(curvature) if abs(curvature) > 1e-6 else p.v_max
        v_stop = math.sqrt(2.0 * p.a_dec * max(remaining - 0.5 * p.goal_tol, 0.0))
        v = min(p.v_max, v_curve, v_stop) * max(math.cos(bearing), 0.0)
        return v, max(-p.w_max, min(p.w_max, v * curvature))
