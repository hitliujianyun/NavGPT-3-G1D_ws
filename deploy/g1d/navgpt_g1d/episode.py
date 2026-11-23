"""One VLN episode on the robot: overlapped inference, smooth tracking, safety.

    control thread    20 Hz: odometry -> safety -> Navigator.control -> authority -> /cmd_vel_no_limit
    inference thread  back to back: newest frame + capture-time pose -> VLA /act -> splice

The robot object supplies ``odom`` (PoseBuffer), ``frame()``, ``scan()`` and
``cmd`` (CmdVelStreamer); robot.G1DRobot is the real one, sim.SimRobot the
simulated one. The VLA client supplies ``reset()`` and ``act(...)``.

A protective stop (obstacle in the stop zone, stale scan) halts the base at
once while inference continues; once the zone has been clear for
``safety.resume_clear_s`` a fresh grant is issued and only predictions from
frames captured after that moment are used.
"""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .authority import MotionAuthority
from .config import AppConfig
from .navigator import Navigator
from .safety import ScanSafety


@dataclass
class EpisodeResult:
    status: str               # arrived | stopped | timeout | stuck | odom_lost | error
    reason: str
    instruction: str
    duration_s: float
    distance_m: float
    requests: int
    replies: int
    protective_stops: int
    mean_latency_s: Optional[float]
    final_pose: Optional[tuple]
    log_dir: str


class EventLog:
    def __init__(self, root: str, save_frames: bool):
        self.dir = Path(root) / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_frames = save_frames
        if save_frames:
            (self.dir / "frames").mkdir(exist_ok=True)
        self._f = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: str, **fields) -> None:
        line = json.dumps({"t": round(time.monotonic(), 4), "event": event, **fields}, default=_jsonable)
        with self._lock:
            self._f.write(line + "\n")

    def frame(self, seq: int, rgb) -> None:
        if not self.save_frames or not hasattr(rgb, "shape") or getattr(rgb, "ndim", 0) != 3:
            return
        try:
            import cv2

            cv2.imwrite(str(self.dir / "frames" / "{:05d}.jpg".format(seq)), rgb[:, :, ::-1])
        except Exception:  # noqa: BLE001 - frames are a convenience
            pass

    def close(self) -> None:
        with self._lock:
            self._f.close()


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class EpisodeRunner:
    def __init__(self, cfg: AppConfig, robot, vla, detect_stuck: bool = True):
        self.cfg, self.robot, self.vla = cfg, robot, vla
        self.detect_stuck = detect_stuck
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status: dict = {"state": "idle"}
        self._status_lock = threading.Lock()
        self.result: Optional[EpisodeResult] = None

    # ── public API ──

    def start(self, instruction: str) -> None:
        if self.running:
            raise RuntimeError("an episode is already running")
        self.result = None
        self._thread = threading.Thread(target=self.run, args=(instruction,), daemon=True)
        self._thread.start()

    def stop(self, reason: str = "stop requested") -> None:
        self._stop_reason = reason
        self._stop.set()

    def wait(self, timeout: Optional[float] = None) -> Optional[EpisodeResult]:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.result

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        with self._status_lock:
            out = dict(self._status)
        if self.result is not None:
            out["result"] = asdict(self.result)
        return out

    def _set_status(self, **fields) -> None:
        with self._status_lock:
            self._status.update(fields)

    # ── episode ──

    def run(self, instruction: str) -> EpisodeResult:
        cfg, robot = self.cfg, self.robot
        ep = cfg.episode
        self._stop.clear()
        self._stop_reason = ""
        log = EventLog(ep.log_dir, ep.save_frames)
        nav = Navigator(cfg.navigator)
        safety = ScanSafety(cfg.safety, cfg.tracker.v_max)
        authority = MotionAuthority()
        lock = threading.Lock()
        stats = {"requests": 0, "latency": [], "errors": 0, "fatal": None, "protective_stops": 0}
        t0 = time.monotonic()
        self._set_status(state="starting", instruction=instruction, log_dir=str(log.dir), started=time.time())
        log.write("start", instruction=instruction, config=asdict(cfg))

        status, reason = "error", ""
        distance, last_pose = 0.0, None
        worker = None
        try:
            if not self._wait_for_odom(t0 + 3.0):
                raise RuntimeError("no odometry on {}".format(cfg.topics.odom))
            self.vla.reset()
            robot.cmd.activate()
            grant = authority.grant("vla_rollout")
            worker = threading.Thread(target=self._inference_loop, daemon=True,
                                      args=(nav, lock, authority, instruction, stats, log))
            worker.start()
            self._set_status(state="running")

            period = 1.0 / ep.control_hz
            t_next = t_last = time.monotonic()
            clear_since = None
            motion_ref = None
            resumed_at_reply = None
            while True:
                now = time.monotonic()
                dt = min(max(now - t_last, 0.2 * period), 3.0 * period)
                t_last = now
                if self._stop.is_set():
                    status, reason = "stopped", self._stop_reason
                    break
                if now - t0 > ep.max_time_s:
                    status, reason = "timeout", "episode time limit"
                    break
                if stats["fatal"]:
                    status, reason = "error", stats["fatal"]
                    break

                latest = robot.odom.latest()
                if latest is None or now - latest[0] > ep.odom_timeout_s:
                    robot.cmd.set(0.0, 0.0)
                    status, reason = "odom_lost", "odometry older than {:.1f} s".format(ep.odom_timeout_s)
                    break
                pose = robot.odom.predict(now)
                if last_pose is not None:
                    distance += math.hypot(pose[0] - last_pose[0], pose[1] - last_pose[1])
                last_pose = pose

                verdict = safety.check(robot.scan(), now, nav.tracker.v, nav.tracker.w)
                if verdict.stop:
                    if not authority.stopped:
                        authority.protective_stop(verdict.reason)
                        stats["protective_stops"] += 1
                        with lock:
                            nav.halt()
                        log.write("protective_stop", reason=verdict.reason, nearest_m=verdict.nearest_m)
                    clear_since = None
                    v = w = 0.0
                elif authority.stopped:
                    v = w = 0.0
                    if safety.clear_for_resume(robot.scan(), now):
                        clear_since = clear_since if clear_since is not None else now
                        if now - clear_since >= cfg.safety.resume_clear_s:
                            authority.clear_protective_stop()
                            grant = authority.grant("vla_rollout")
                            with lock:
                                nav.resume(now)
                                resumed_at_reply = nav.replies
                            log.write("resume")
                            clear_since = None
                    else:
                        clear_since = None
                else:
                    with lock:
                        v, w = nav.control(pose, now, dt, v_cap=verdict.v_cap)
                        finished = nav.finished
                    v, w = authority.filter(v, w, grant)
                    if finished:
                        status, reason = "arrived", "VLA arrival confirmed"
                        robot.cmd.set(0.0, 0.0)
                        break

                robot.cmd.set(v, w)
                robot.odom.set_command(v, w)
                if resumed_at_reply is not None and nav.replies > resumed_at_reply:
                    resumed_at_reply = None
                if authority.stopped:
                    phase = "protective_stop"
                elif resumed_at_reply is not None:
                    phase = "resuming"
                elif verdict.v_cap is not None:
                    phase = "obstacle_slow"
                else:
                    phase = "running"
                log.write("cmd", v=round(v, 5), w=round(w, 5), dt=round(dt, 5), pose=[round(c, 4) for c in pose],
                          v_cap=verdict.v_cap, phase=phase)

                motion_ref, stuck = self._stuck_check(motion_ref, now, pose, v, w)
                if stuck:
                    status, reason = "stuck", "commanded motion without odometry change"
                    break

                self._set_status(pose=pose, v=v, w=w, elapsed_s=now - t0, replies=nav.replies,
                                 requests=stats["requests"], protective_stop=authority.stopped,
                                 stop_reason=authority.stop_reason if authority.stopped else "",
                                 final=nav.tracker.final, distance_m=distance)
                t_next += period
                late = time.monotonic() - t_next
                if late > period:
                    t_next += late
                time.sleep(max(0.0, t_next - time.monotonic()))
        except Exception as exc:  # noqa: BLE001 - the base must still be released
            status, reason = "error", repr(exc)
            log.write("exception", error=traceback.format_exc())
        finally:
            self._stop.set()
            authority.revoke(status)
            robot.cmd.set(0.0, 0.0)
            if worker is not None:
                worker.join(timeout=cfg.vla.timeout_s + 1.0)
            robot.cmd.release(cfg.base.zero_hold_s)

        lat = stats["latency"]
        self.result = EpisodeResult(
            status=status, reason=reason, instruction=instruction,
            duration_s=round(time.monotonic() - t0, 2), distance_m=round(distance, 3),
            requests=stats["requests"], replies=nav.replies, protective_stops=stats["protective_stops"],
            mean_latency_s=round(sum(lat) / len(lat), 3) if lat else None,
            final_pose=last_pose, log_dir=str(log.dir))
        log.write("result", **asdict(self.result))
        log.close()
        self._set_status(state="finished")
        return self.result

    def _wait_for_odom(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            latest = self.robot.odom.latest()
            if latest is not None and time.monotonic() - latest[0] < self.cfg.episode.odom_timeout_s:
                return True
            time.sleep(0.05)
        return False

    def _stuck_check(self, ref, now, pose, v, w):
        ep = self.cfg.episode
        if not self.detect_stuck or (abs(v) < 0.05 and abs(w) < 0.1):
            return None, False
        if ref is None:
            return (now, pose), False
        t_ref, p_ref = ref
        moved = math.hypot(pose[0] - p_ref[0], pose[1] - p_ref[1]) > ep.stuck_dist_m
        turned = abs(math.remainder(pose[2] - p_ref[2], 2 * math.pi)) > 0.1
        if moved or turned:
            return (now, pose), False
        return ref, now - t_ref > ep.stuck_time_s

    def _inference_loop(self, nav: Navigator, lock, authority, instruction: str, stats: dict, log) -> None:
        cfg, robot = self.cfg, self.robot
        failures = 0
        while not self._stop.is_set():
            t_start = time.monotonic()
            if stats["requests"] >= cfg.episode.max_requests:
                stats["fatal"] = "request limit reached"
                return
            frame = robot.frame()
            pose = robot.odom.at(frame[1] - cfg.camera.latency_s) if frame else None
            if frame is None or pose is None:
                time.sleep(0.02)
                continue
            rgb, t_rx = frame
            with lock:
                req = nav.next_request(t_rx - cfg.camera.latency_s, pose)
            stats["requests"] += 1
            try:
                reply = self.vla.act(rgb, instruction, commit=req.commit, seq=req.seq)
                failures = 0
            except Exception as exc:  # noqa: BLE001 - keep driving on the current path
                failures += 1
                log.write("act_failed", seq=req.seq, commit=req.commit, error=repr(exc))
                if failures >= 10:
                    stats["fatal"] = "VLA unreachable: {!r}".format(exc)
                    return
                time.sleep(0.2)
                continue
            if req.commit:
                log.frame(req.seq, rgb)
            now = time.monotonic()
            with lock:
                accepted = nav.on_reply(req, reply["waypoints"], robot.odom.predict(now), now)
                final, deviation = nav.tracker.final, nav.last_deviation
            stats["latency"].append(now - req.t_capture)
            log.write("reply", seq=req.seq, commit=req.commit, accepted=accepted,
                      latency_s=round(now - req.t_capture, 3), infer_s=reply.get("infer_s"),
                      frames=reply.get("frames"), deviation_m=round(deviation, 3), final=final,
                      capture_pose=req.pose, waypoints=reply["waypoints"])
            if final:
                return
            wait = cfg.episode.min_request_period_s - (time.monotonic() - t_start)
            if wait > 0:
                self._stop.wait(wait)
