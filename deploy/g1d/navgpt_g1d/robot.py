"""The real G1-D: DDS odometry/scan/commands plus the teleimager head camera.

base.output selects who drives the chassis:

* direct: this process is the only writer of rt/cmd_vel_no_limit (checked at
  start-up), and watches the remote sticks and chassis bumper/cliff events;
* guard: commands go to rt/nav/cmd_vel for the g1d_ws / g1d_ros2_ws base guard,
  which owns rt/cmd_vel_no_limit; its rt/nav/guard_state drives supervision.
"""

from __future__ import annotations

import logging
import time

from .config import AppConfig
from .dds import (CmdVelStreamer, GuardStateSubscriber, JoystickSubscriber, OdomSubscriber, ScanSubscriber,
                  dds_twist_writer, find_writers, init_dds)
from .pose_buffer import PoseBuffer
from .supervisor import CLEAR, EventLatch, GuardState, JoystickMonitor, Supervision, combine

log = logging.getLogger("navgpt_g1d.robot")


class PreflightError(RuntimeError):
    pass


class G1DRobot:
    def __init__(self, cfg: AppConfig, dry_run: bool = False, with_camera: bool = True):
        from .camera import HeadCamera, resolve_camera

        init_dds(cfg.dds.domain_id, cfg.dds.interface, cfg.dds.interface_ip)
        self.cfg, self.dry_run = cfg, dry_run
        self.output = cfg.base.output
        sup = cfg.supervisor

        self.odom = PoseBuffer()
        self._odom = OdomSubscriber(self.odom, cfg.topics.odom, cfg.base.odom_yaw_sign)
        self._scan = ScanSubscriber(cfg.topics.scan) if cfg.safety.enabled else None

        self.guard = None
        self.joystick = None
        self.events = None
        self._poller = None
        if self.output == "guard":
            self.guard = GuardState(sup.guard_state_timeout_s)
            self._guard_sub = GuardStateSubscriber(self.guard, cfg.topics.guard_state)
        else:
            if sup.joystick:
                self.joystick = JoystickMonitor(sup.joystick_deadzone)
                self._joy_sub = JoystickSubscriber(self.joystick, cfg.topics.joystick)
            if sup.chassis_http:
                from .chassis_events import ChassisEventPoller

                self.events = EventLatch(sup.estop_events)
                self._poller = ChassisEventPoller(sup.chassis_http, self.events, sup.event_poll_s)

        self.camera = None
        self.camera_info = {}
        if with_camera:
            cam = cfg.camera
            port, binocular = resolve_camera(cam.host, cam.port, cam.binocular, cam.request_port, cam.name)
            self.camera_info = {"host": cam.host, "port": port, "binocular": binocular, "eye": cam.eye}
            self.camera = HeadCamera(host=cam.host, port=port, binocular=binocular, eye=cam.eye,
                                     out_size=cam.out_size, stale_after_s=cam.stale_after_s)

        self.cmd = self._make_streamer()

    def _make_streamer(self) -> CmdVelStreamer:
        cfg = self.cfg
        if self.dry_run:
            return CmdVelStreamer(lambda v, w: None, cfg.base.publish_hz, cfg.base.cmd_timeout_s)
        if self.output == "guard":
            self._wait_for_guard(3.0)
            return CmdVelStreamer(dds_twist_writer(cfg.topics.guard_cmd), cfg.base.publish_hz, cfg.base.cmd_timeout_s)
        self._check_single_writer()
        return CmdVelStreamer(dds_twist_writer(cfg.topics.cmd_vel), cfg.base.publish_hz, cfg.base.cmd_timeout_s,
                              cfg.base.cmd_x_sign, cfg.base.cmd_yaw_sign)

    def _check_single_writer(self) -> None:
        topic = self.cfg.topics.cmd_vel
        try:
            writers = find_writers(topic, self.cfg.dds.domain_id, 1.5)
        except Exception as exc:  # noqa: BLE001 - discovery is a safeguard, not a requirement
            log.warning("writer discovery on %s unavailable: %r", topic, exc)
            return
        if not writers:
            return
        msg = ("{} already has {} writer(s) (another navigation program or the base guard). "
               "Stop it, or set base.output: guard to drive through the guard.").format(topic, len(writers))
        if self.cfg.base.other_writers == "refuse":
            raise PreflightError(msg)
        log.warning(msg)

    def _wait_for_guard(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.guard.verdict(time.monotonic()).abort_status != "guard_lost":
                return
            time.sleep(0.05)
        raise PreflightError("base.output is guard but no {} arrived within {:.0f} s: start the base guard "
                             "(g1d_ws: g1d-guard; g1d_ros2_ws: robot_interface.launch.py)".format(
                                 self.cfg.topics.guard_state, timeout_s))

    # ── episode interface ──

    def frame(self):
        return self.camera.latest() if self.camera else None

    def scan(self):
        return self._scan.latest() if self._scan else None

    def release_zero_hold(self) -> float:
        return self.cfg.base.guard_release_s if self.output == "guard" else self.cfg.base.zero_hold_s

    def supervise(self, now: float, since: float) -> Supervision:
        """Abort / blocked verdict for an episode that started at ``since``."""
        if self.guard is not None:
            return self.guard.verdict(now)
        verdicts = []
        if self.events is not None and self.events.latched:
            verdicts.append(Supervision("estop", "chassis event {}".format(self.events.latched)))
        if self.joystick is not None and self.joystick.active_since(since):
            verdicts.append(Supervision("manual_override", "remote sticks moved"))
        return combine(*verdicts) if verdicts else CLEAR

    def clear_estop(self) -> None:
        if self.events is not None:
            self.events.clear()

    def health(self) -> dict:
        latest = self.odom.latest()
        out = {
            "output": self.output,
            "odom_hz": round(self._odom.rate.hz, 1),
            "odom_pose": latest[1] if latest else None,
            "odom_age_s": round(time.monotonic() - latest[0], 3) if latest else None,
            "scan_hz": round(self._scan.rate.hz, 1) if self._scan else None,
            "camera": self.camera_info,
            "camera_hz": round(self.camera.rate_hz, 1) if self.camera else None,
            "cmd_active": self.cmd.active,
            "dry_run": self.dry_run,
        }
        if self.guard is not None:
            state, t = self.guard.latest()
            out["guard"] = None if state is None else {k: state.get(k) for k in (
                "blocked", "estop", "estop_reason", "manual", "manual_source", "front_clear", "publishing")}
            out["guard_age_s"] = None if state is None else round(time.monotonic() - t, 3)
        if self.joystick is not None:
            out["joystick_samples"] = self.joystick.samples
        if self._poller is not None:
            out["chassis_http_ok"] = self._poller.ok
            out["chassis_estop"] = self.events.latched
        return out

    def close(self) -> None:
        self.cmd.close()
        if self.camera:
            self.camera.close()
        if self._poller:
            self._poller.close()
