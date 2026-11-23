"""Planar poses (x, y, yaw) in the odometry frame and robot-frame conversions."""

from __future__ import annotations

import math
from typing import Tuple

Pose = Tuple[float, float, float]


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def body_to_world(pose: Pose, x: float, y: float, theta: float = 0.0) -> Pose:
    """Robot-frame offset (+x forward, +y left, +theta CCW) composed onto a world pose."""
    px, py, pyaw = pose
    c, s = math.cos(pyaw), math.sin(pyaw)
    return px + c * x - s * y, py + s * x + c * y, wrap_angle(pyaw + theta)


def world_to_body(pose: Pose, wx: float, wy: float) -> tuple[float, float]:
    px, py, pyaw = pose
    dx, dy = wx - px, wy - py
    c, s = math.cos(pyaw), math.sin(pyaw)
    return c * dx + s * dy, -s * dx + c * dy


def integrate_unicycle(pose: Pose, v: float, w: float, dt: float) -> Pose:
    x, y, yaw = pose
    if abs(w) < 1e-9:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    r = v / w
    yaw2 = yaw + w * dt
    return x + r * (math.sin(yaw2) - math.sin(yaw)), y - r * (math.cos(yaw2) - math.cos(yaw)), wrap_angle(yaw2)
