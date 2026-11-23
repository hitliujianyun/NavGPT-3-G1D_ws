"""Inference/execution overlap, without threads (episode.py adds them).

Two loops share one Navigator:

* the inference loop calls ``next_request`` with the newest frame's capture
  time and pose, sends it to the VLA, and passes the reply to ``on_reply``;
* the control loop calls ``control`` at 20 Hz and never waits for the VLA.

Continuity between predictions comes from anchoring each one at its capture
pose and splicing it onto the tracked path (path.splice). History spacing stays
close to training (0.25 m forward steps, 15 deg turns): a frame is committed to
the VLA's visual history only after the robot moved ``keyframe_m`` or turned
``keyframe_rad`` since the last committed frame; other requests use the frame as
the current observation without keeping it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .geometry import Pose, wrap_angle
from .path import RefPath, deviation, path_from_waypoints, splice
from .tracker import PathTracker, TrackerParams

LAST_WP_STOP_MAGNITUDE = 0.25
LAST_WP_STOP_THETA = math.pi / 6


def suggests_arrival(waypoints) -> bool:
    """The VLA never emits STOP; a collapsed horizon is its implicit arrival signal."""
    if not waypoints:
        return False
    x, y, theta = (float(v) for v in waypoints[-1][:3])
    return math.hypot(x, y) < LAST_WP_STOP_MAGNITUDE and abs(theta) < LAST_WP_STOP_THETA


@dataclass(frozen=True)
class NavigatorParams:
    blend_m: float = 0.5           # >= tracker lookahead, so commands see the blend
    ds: float = 0.05
    keyframe_m: float = 0.25
    keyframe_rad: float = math.radians(15)
    max_reply_age_s: float = 3.0   # replies to frames older than this are discarded
    starve_s: float = 1.5          # no reply accepted for this long: slow down
    stop_s: float = 3.0            # ...and for this long: stop
    starve_speed: float = 0.1      # m/s cap while starved
    arrival_hits: int = 2
    tracker: TrackerParams = field(default_factory=TrackerParams)


@dataclass(frozen=True)
class Request:
    seq: int
    t_capture: float
    pose: Pose
    commit: bool


class Navigator:
    def __init__(self, params: NavigatorParams = NavigatorParams()):
        self.p = params
        self.tracker = PathTracker(params.tracker)
        self.path: Optional[RefPath] = None
        self.finished = False
        self.last_reply_time = -math.inf
        self.last_deviation = 0.0
        self.replies = 0
        self._seq = 0
        self._applied_seq = -1
        self._min_capture_t = -math.inf
        self._last_commit_pose: Optional[Pose] = None
        self._arrival_streak = 0

    # ── inference side ──

    def next_request(self, t_capture: float, pose: Pose) -> Request:
        commit = self._last_commit_pose is None or self._moved_enough(pose)
        if commit:
            self._last_commit_pose = pose
        self._seq += 1
        return Request(self._seq, t_capture, pose, commit)

    def _moved_enough(self, pose: Pose) -> bool:
        x, y, yaw = self._last_commit_pose
        return (math.hypot(pose[0] - x, pose[1] - y) >= self.p.keyframe_m
                or abs(wrap_angle(pose[2] - yaw)) >= self.p.keyframe_rad)

    def on_reply(self, req: Request, waypoints, robot_pose: Pose, now: float) -> bool:
        """Splice a prediction onto the tracked path; False if it was stale."""
        if (self.finished or req.seq <= self._applied_seq or req.t_capture < self._min_capture_t
                or now - req.t_capture > self.p.max_reply_age_s):
            return False
        self._applied_seq, self.last_reply_time = req.seq, now
        self.replies += 1

        self._arrival_streak = self._arrival_streak + 1 if suggests_arrival(waypoints) else 0
        final = self._arrival_streak >= self.p.arrival_hits

        new = path_from_waypoints(req.pose, waypoints, self.p.ds, req.t_capture)
        if self.path is not None:
            self.last_deviation = deviation(self.path, new, robot_pose[:2])
        self.path = splice(self.path, new, robot_pose[:2], self.p.blend_m, self.p.ds)
        self.tracker.set_path(self.path, final=final)
        return True

    # ── control side ──

    def control(self, pose: Pose, now: float, dt: float, v_cap: Optional[float] = None) -> Tuple[float, float]:
        """One control tick; v_cap is an external speed limit (e.g. from the safety monitor)."""
        if self.finished:
            return 0.0, 0.0
        age = now - self.last_reply_time
        if age > self.p.stop_s and not self.tracker.final:
            self.tracker.stop()
        if age > self.p.starve_s and not self.tracker.final:
            v_cap = self.p.starve_speed if v_cap is None else min(v_cap, self.p.starve_speed)
        v, w = self.tracker.step(pose, dt, v_cap=v_cap)
        if self.tracker.done:
            self.finished = True
        return v, w

    def halt(self) -> None:
        """Zero the commanded velocity at once (protective stop); keep the path."""
        self.tracker.v = self.tracker.w = 0.0

    def resume(self, now: float) -> None:
        """After a protective stop: only predictions from frames captured from now on count."""
        self.halt()
        self._min_capture_t = now
        self.last_reply_time = now
        self._arrival_streak = 0
        if not self.tracker.final:
            self.tracker.stop()
