"""The real G1-D: DDS odometry/scan/cmd_vel plus the teleimager head camera."""

from __future__ import annotations

import time

from .config import AppConfig
from .dds import CmdVelStreamer, OdomSubscriber, ScanSubscriber, dds_twist_writer, init_dds
from .pose_buffer import PoseBuffer


class G1DRobot:
    def __init__(self, cfg: AppConfig, dry_run: bool = False):
        from .camera import HeadCamera

        init_dds(cfg.dds.domain_id, cfg.dds.interface, cfg.dds.interface_ip)
        self.cfg, self.dry_run = cfg, dry_run
        self.odom = PoseBuffer()
        self._odom = OdomSubscriber(self.odom, cfg.topics.odom, cfg.base.odom_yaw_sign)
        self._scan = ScanSubscriber(cfg.topics.scan) if cfg.safety.enabled else None
        self.camera = HeadCamera(host=cfg.camera.host, port=cfg.camera.port, binocular=cfg.camera.binocular,
                                 eye=cfg.camera.eye, out_size=cfg.camera.out_size,
                                 stale_after_s=cfg.camera.stale_after_s)
        write = (lambda v, w: None) if dry_run else dds_twist_writer(cfg.topics.cmd_vel)
        self.cmd = CmdVelStreamer(write, cfg.base.publish_hz, cfg.base.cmd_timeout_s,
                                  cfg.base.cmd_x_sign, cfg.base.cmd_yaw_sign)

    def frame(self):
        return self.camera.latest()

    def scan(self):
        return self._scan.latest() if self._scan else None

    def health(self) -> dict:
        latest = self.odom.latest()
        return {
            "odom_hz": round(self._odom.rate.hz, 1),
            "odom_pose": latest[1] if latest else None,
            "odom_age_s": round(time.monotonic() - latest[0], 3) if latest else None,
            "scan_hz": round(self._scan.rate.hz, 1) if self._scan else None,
            "camera_hz": round(self.camera.rate_hz, 1),
            "cmd_active": self.cmd.active,
            "dry_run": self.dry_run,
        }

    def close(self) -> None:
        self.cmd.close()
        self.camera.close()
