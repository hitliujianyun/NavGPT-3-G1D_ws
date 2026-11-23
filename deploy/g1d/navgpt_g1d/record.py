"""Teleoperated demonstrations for navgpt_sft (train/), recorded on the robot.

The operator drives with the remote; nothing is commanded from here. At
``record.rate_hz`` the head-camera frame (the exact 400x400 left-eye crop the
VLA sees in deployment) is saved with the odometry pose at its capture time.
Each episode becomes one manifest line in navgpt_sft's format::

    {"episode_id": ..., "instructions": [...], "frames": [{"image": "<id>/00000.jpg", "pose": [x, y, yaw]}, ...]}

Keyframes and the eight-waypoint targets are derived at training time, so the
frames are kept dense.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional


def _write_jpeg(path: Path, rgb, quality: int) -> None:
    try:
        import cv2

        if not cv2.imwrite(str(path), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise OSError("cv2.imwrite failed for {}".format(path))
    except ImportError:
        from PIL import Image

        Image.fromarray(rgb).save(path, quality=quality)


class EpisodeRecorder:
    """Collects frames for one episode; ``finish`` writes the images' manifest line."""

    def __init__(self, robot, out_dir: str, latency_s: float, rate_hz: float, jpeg_quality: int = 95,
                 episode_id: Optional[str] = None):
        self.robot = robot
        self.root = Path(out_dir)
        self.latency_s, self.period = latency_s, 1.0 / rate_hz
        self.quality = jpeg_quality
        self.episode_id = episode_id or time.strftime("g1d_%Y%m%d_%H%M%S")
        self.dir = self.root / self.episode_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames: List[dict] = []
        self.skipped = 0
        self._last_rx = -math.inf

    def poll(self) -> bool:
        """Save the newest frame if it is new and its capture-time pose is known."""
        frame = self.robot.frame()
        if frame is None:
            self.skipped += 1
            return False
        rgb, t_rx = frame
        if t_rx <= self._last_rx:
            return False
        pose = self.robot.odom.at(t_rx - self.latency_s)
        latest = self.robot.odom.latest()
        if pose is None or latest is None or t_rx - latest[0] > 0.5:
            self.skipped += 1                       # no odometry around this frame
            return False
        self._last_rx = t_rx
        name = "{:05d}.jpg".format(len(self.frames))
        _write_jpeg(self.dir / name, rgb, self.quality)
        self.frames.append({"image": "{}/{}".format(self.episode_id, name),
                            "pose": [round(float(c), 5) for c in pose], "t": round(t_rx, 4)})
        return True

    def run(self, stop, max_s: float = 600.0) -> int:
        """Record until ``stop()`` is true (or max_s), at the configured rate."""
        t_next = time.monotonic()
        t_end = t_next + max_s
        while not stop() and time.monotonic() < t_end:
            self.poll()
            t_next += self.period
            time.sleep(max(0.0, t_next - time.monotonic()))
        return len(self.frames)

    def path_length(self) -> float:
        p = [f["pose"] for f in self.frames]
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:]))

    def finish(self, instructions: List[str], manifest: str, **meta) -> Optional[dict]:
        """Append the episode to the manifest; None (and nothing written) if it has < 2 frames."""
        instructions = [s.strip() for s in instructions if s and s.strip()]
        if len(self.frames) < 2 or not instructions:
            return None
        record = {"episode_id": self.episode_id, "instructions": instructions,
                  "frames": [{"image": f["image"], "pose": f["pose"]} for f in self.frames],
                  "source": "g1d_teleop", "path_m": round(self.path_length(), 3), **meta}
        with open(self.root / manifest, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
