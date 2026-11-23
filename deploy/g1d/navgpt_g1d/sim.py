"""Simulated G1-D for end-to-end runs without the robot.

SimRobot has the same surface as robot.G1DRobot (odom, frame(), scan(), cmd),
so EpisodeRunner runs unchanged, in real time, with all its threads:

* the base follows /cmd_vel with a first-order lag, integrated at 100 Hz;
* odometry is published at 10 Hz with optional noise;
* a 360-beam scan sees circular obstacles, which may move (a person crossing);
* RouteOracleVLA plays the VLA: it looks up the true pose at the frame's
  capture time and predicts eight noisy waypoints along a known route, after a
  random latency with occasional Wi-Fi stalls.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .dds import CmdVelStreamer, RateMeter, Scan
from .geometry import Pose, integrate_unicycle, world_to_body, wrap_angle
from .path import resample
from .pose_buffer import PoseBuffer

ROUTES = {
    "straight": [(0.0, 0.0), (4.0, 0.0)],
    "L": [(0.0, 0.0), (3.0, 0.0), (3.0, 2.5)],
    "S": [(0.0, 0.0), (1.5, 0.0), (2.5, 1.0), (2.5, 2.0), (4.0, 3.0)],
}


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = 0.25
    vx: float = 0.0
    vy: float = 0.0
    t_on: float = 0.0           # seconds after the world starts
    t_off: float = math.inf

    def centre(self, t: float):
        if not (self.t_on <= t <= self.t_off):
            return None
        dt = t - self.t_on
        return self.x + self.vx * dt, self.y + self.vy * dt


class SimFrame:
    """Stands in for an image: carries its capture time to the oracle VLA."""

    def __init__(self, t: float):
        self.t = t


class SimWorld:
    def __init__(self, start: Pose = (0.0, 0.0, 0.0), base_tau_s: float = 0.15, odom_hz: float = 10.0,
                 odom_noise_m: float = 0.0, obstacles: Sequence[Obstacle] = (), seed: int = 0):
        self.t0 = time.monotonic()
        self._pose = start
        self._v = self._w = 0.0
        self._target = (0.0, 0.0)
        self._tau = base_tau_s
        self._history: deque = deque(maxlen=2000)
        self._lock = threading.Lock()
        self.obstacles = list(obstacles)
        self.odom = PoseBuffer()
        self.odom_rate = RateMeter()
        self._odom_period = 1.0 / odom_hz
        self._noise = odom_noise_m
        self._rng = random.Random(seed)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def apply(self, v: float, w: float) -> None:
        with self._lock:
            self._target = (v, w)

    def pose_at(self, t: float) -> Pose:
        with self._lock:
            hist = list(self._history)
        best = min(hist, key=lambda h: abs(h[0] - t)) if hist else (t, self._pose)
        return best[1]

    @property
    def pose(self) -> Pose:
        with self._lock:
            return self._pose

    def _loop(self) -> None:
        dt, t_next, t_odom = 0.01, time.monotonic(), 0.0
        while self._running:
            now = time.monotonic()
            with self._lock:
                k = 1.0 - math.exp(-dt / self._tau)
                self._v += k * (self._target[0] - self._v)
                self._w += k * (self._target[1] - self._w)
                self._pose = integrate_unicycle(self._pose, self._v, self._w, dt)
                self._history.append((now, self._pose))
                pose = self._pose
            if now - t_odom >= self._odom_period:
                t_odom = now
                n = self._noise
                self.odom.add(now, (pose[0] + self._rng.gauss(0, n), pose[1] + self._rng.gauss(0, n), pose[2]))
                self.odom_rate.tick(now)
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))

    def scan(self, beams: int = 360) -> Scan:
        now = time.monotonic()
        pose = self.pose
        angles = np.linspace(-math.pi, math.pi, beams, endpoint=False)
        best = np.full(beams, np.inf)
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        for ob in self.obstacles:
            c = ob.centre(now - self.t0)
            if c is None:
                continue
            cb = np.array(world_to_body(pose, *c))
            proj = d @ cb
            disc = proj ** 2 - (cb @ cb - ob.radius ** 2)
            hit = np.where(disc >= 0, proj - np.sqrt(np.maximum(disc, 0)), np.inf)
            best = np.minimum(best, np.where(hit > 0, hit, np.inf))
        ok = np.isfinite(best)
        return Scan(now, angles[ok].astype(np.float32), best[ok].astype(np.float32))

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


class SimRobot:
    def __init__(self, world: SimWorld, publish_hz: float = 30.0, cmd_timeout_s: float = 0.3):
        self.world = world
        self.odom = world.odom
        self.cmd = CmdVelStreamer(world.apply, publish_hz, cmd_timeout_s)
        self.dry_run = False

    def frame(self):
        now = time.monotonic()
        return SimFrame(now), now

    def scan(self) -> Scan:
        return self.world.scan()

    def health(self) -> dict:
        return {"odom_hz": round(self.world.odom_rate.hz, 1), "odom_pose": self.world.pose,
                "scan_hz": None, "camera_hz": None, "cmd_active": self.cmd.active, "sim": True}

    def close(self) -> None:
        self.cmd.close()
        self.world.close()


class RouteOracleVLA:
    def __init__(self, world: SimWorld, route: List[tuple], latency_s=(0.3, 0.9), stall_prob: float = 0.05,
                 stall_s: float = 2.0, lateral_noise_m: float = 0.04, spacing_m: float = 0.15, seed: int = 0):
        d = np.diff(np.asarray(route, dtype=float), axis=0)[-1]
        self.route = resample(route, math.atan2(d[1], d[0]), 0.02)
        self.world = world
        self.latency_s, self.stall_prob, self.stall_s = latency_s, stall_prob, stall_s
        self.noise, self.spacing = lateral_noise_m, spacing_m
        self._rng = random.Random(seed)
        self.frames = 0

    def info(self) -> dict:
        return {"model": "route-oracle", "view_num": 1}

    def reset(self, seed: Optional[int] = None) -> None:
        self.frames = 0

    def act(self, rgb, instruction: str, commit: bool = True, seq=None) -> dict:
        pose = self.world.pose_at(rgb.t)
        rng = self._rng
        delay = self.stall_s if rng.random() < self.stall_prob else rng.uniform(*self.latency_s)
        time.sleep(delay)
        self.frames += int(commit)
        s0 = self.route.project(pose[:2])
        if self.route.length - s0 < 0.3:
            wps = [[0.02, 0.0, 0.0]] * 8
        else:
            wps = []
            for k in range(1, 9):
                s = s0 + self.spacing * k
                bx, by = world_to_body(pose, *self.route.point(s))
                theta = wrap_angle(self.route.heading(s) - pose[2])
                wps.append([bx + rng.gauss(0, 0.02), by + rng.gauss(0, self.noise), theta + rng.gauss(0, 0.05)])
        return {"waypoints": wps, "infer_s": delay, "frames": self.frames, "seq": seq}
