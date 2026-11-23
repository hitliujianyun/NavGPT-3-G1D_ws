"""Short odometry history, so a prediction can be anchored at its image's capture time.

Odometry arrives at about 10 Hz while the controller runs at 20 Hz and replies
come back 0.3-2 s after capture over Wi-Fi. ``at(t)`` interpolates the pose at
any recent time; ``predict(now)`` dead-reckons the latest pose forward with the
last commanded velocity so the controller never acts on a 100 ms old pose.
"""

from __future__ import annotations

import bisect
import threading
from collections import deque

from .geometry import Pose, integrate_unicycle, wrap_angle


class PoseBuffer:
    def __init__(self, horizon_s: float = 5.0):
        self._horizon_s = horizon_s
        self._lock = threading.Lock()
        self._t: deque[float] = deque()
        self._p: deque[Pose] = deque()
        self._cmd = (0.0, 0.0)

    def add(self, t: float, pose: Pose) -> None:
        with self._lock:
            if self._t and t <= self._t[-1]:
                return
            self._t.append(t)
            self._p.append(pose)
            while self._t and self._t[0] < t - self._horizon_s:
                self._t.popleft()
                self._p.popleft()

    def set_command(self, v: float, w: float) -> None:
        with self._lock:
            self._cmd = (v, w)

    def latest(self) -> tuple[float, Pose] | None:
        with self._lock:
            return (self._t[-1], self._p[-1]) if self._t else None

    def at(self, t: float) -> Pose | None:
        """Pose at time t, interpolated; clamped to the oldest/newest sample."""
        with self._lock:
            if not self._t:
                return None
            ts = list(self._t)
            i = bisect.bisect_left(ts, t)
            if i <= 0:
                return self._p[0]
            if i >= len(ts):
                return self._p[-1]
            t0, t1 = ts[i - 1], ts[i]
            (x0, y0, a0), (x1, y1, a1) = self._p[i - 1], self._p[i]
        u = (t - t0) / (t1 - t0)
        return x0 + u * (x1 - x0), y0 + u * (y1 - y0), wrap_angle(a0 + u * wrap_angle(a1 - a0))

    def predict(self, now: float, max_extrapolation_s: float = 0.3) -> Pose | None:
        """Latest pose, dead-reckoned to ``now`` with the last command."""
        with self._lock:
            if not self._t:
                return None
            t, pose = self._t[-1], self._p[-1]
            v, w = self._cmd
        return integrate_unicycle(pose, v, w, min(max(now - t, 0.0), max_extrapolation_s))
