"""Signals from outside the episode that override it: guard, remote, chassis events.

Two output modes share one verdict (``Supervision``):

* guard mode (base.output: guard): the g1d_ws / g1d_ros2_ws base guard is the
  sole writer of rt/cmd_vel_no_limit and already handles the reflex stop,
  bumper/cliff e-stop and remote takeover; its rt/nav/guard_state is read here.
* direct mode: navgpt_g1d writes the base itself, so the remote sticks
  (rt/wirelesscontroller) and chassis HTTP events are watched here.

``abort`` ends the episode (the operator or a latched e-stop took over);
``blocked`` means the guard is holding the robot back (obstacle ahead): the
episode keeps predicting and tracking, and only stuck detection is suspended,
since the guard releases on its own once the way is clear.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Supervision:
    abort_status: str = ""        # manual_override | estop | guard_lost
    abort_reason: str = ""
    blocked: str = ""             # guard's blocked reason (front, rear, rotate, scan_stale, ...)

    @property
    def abort(self) -> bool:
        return bool(self.abort_status)


CLEAR = Supervision()


class GuardState:
    """Latest rt/nav/guard_state JSON, with its receipt time."""

    def __init__(self, timeout_s: float = 0.5):
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._state: Optional[dict] = None
        self._t = -math.inf

    def update(self, text: str, now: Optional[float] = None) -> None:
        try:
            state = json.loads(text)
        except ValueError:
            return
        if isinstance(state, dict):
            with self._lock:
                self._state, self._t = state, time.monotonic() if now is None else now

    def latest(self):
        with self._lock:
            return self._state, self._t

    def verdict(self, now: float) -> Supervision:
        state, t = self.latest()
        if state is None:
            return Supervision("guard_lost", "no rt/nav/guard_state received")
        if now - t > self.timeout_s:
            return Supervision("guard_lost", "rt/nav/guard_state older than {:.1f} s".format(self.timeout_s))
        if state.get("estop"):
            return Supervision("estop", "guard e-stop: {}".format(state.get("estop_reason") or "latched"))
        if state.get("manual"):
            return Supervision("manual_override", "guard: manual control ({})".format(state.get("manual_source") or "remote"))
        return Supervision(blocked=str(state.get("blocked") or ""))


class JoystickMonitor:
    """Remote sticks beyond the deadzone mean the operator is taking over."""

    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self._lock = threading.Lock()
        self._active_t = -math.inf
        self.samples = 0

    def update(self, lx: float, ly: float, rx: float, ry: float, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self.samples += 1
            if max(abs(lx), abs(ly), abs(rx), abs(ry)) > self.deadzone:
                self._active_t = now

    def active_since(self, t0: float) -> bool:
        with self._lock:
            return self._active_t >= t0


class EventLatch:
    """Chassis events (bumper, cliff) seen after start-up latch an e-stop."""

    def __init__(self, stop_types=("BUMPER_TRIGGERED", "CLIFF_DETECTED")):
        self.stop_types = set(stop_types)
        self._seen = None
        self._lock = threading.Lock()
        self.latched = ""

    def update(self, events) -> list:
        keys = {(e.get("type"), e.get("timestamp", e.get("time", e.get("id")))) for e in events if isinstance(e, dict)}
        with self._lock:
            if self._seen is None:          # events from before start-up never latch
                self._seen = keys
                return []
            new = [k[0] for k in keys - self._seen if k[0]]
            self._seen |= keys
            hits = [t for t in new if t in self.stop_types]
            if hits and not self.latched:
                self.latched = hits[0]
            return new

    def clear(self) -> None:
        with self._lock:
            self.latched = ""


def combine(*verdicts: Supervision) -> Supervision:
    """First abort wins (in the order given); otherwise any blocked reason."""
    for v in verdicts:
        if v.abort:
            return v
    for v in verdicts:
        if v.blocked:
            return v
    return CLEAR
