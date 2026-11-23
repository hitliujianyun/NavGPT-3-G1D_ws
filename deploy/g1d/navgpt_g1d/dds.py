"""DDS I/O on the robot LAN: odometry and laser scans in, /cmd_vel_no_limit out.

All samples are stamped with the local monotonic clock on arrival, the clock the
camera frames use too, so poses can be looked up at a frame's capture time.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .pose_buffer import PoseBuffer


def init_dds(domain_id: int = 0, interface: Optional[str] = None, interface_ip: Optional[str] = None) -> None:
    """Initialise the unitree_sdk2py channel factory.

    unitree_sdk2py selects a NIC by name only. With interface_ip the CycloneDDS
    config selects it by address instead, which survives NIC renames and keeps
    DDS off the Wi-Fi interface when both are up.
    """
    import unitree_sdk2py.core.channel as channel

    if interface_ip:
        channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
            'name="$__IF_NAME__$"', 'address="$__IF_NAME__$"')
        channel.ChannelFactoryInitialize(domain_id, interface_ip)
    else:
        channel.ChannelFactoryInitialize(domain_id, interface)


class RateMeter:
    def __init__(self, window: int = 20):
        self._times: list = []
        self._window = window
        self.last = -math.inf

    def tick(self, t: float) -> None:
        self.last = t
        self._times.append(t)
        del self._times[:-self._window]

    @property
    def hz(self) -> float:
        if len(self._times) < 2 or time.monotonic() - self.last > 1.0:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


class OdomSubscriber:
    def __init__(self, buffer: PoseBuffer, topic: str, yaw_sign: float = 1.0):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_

        self.buffer, self.yaw_sign, self.rate = buffer, yaw_sign, RateMeter()
        self._sub = ChannelSubscriber(topic, Odometry_)
        self._sub.Init(self._on_msg, 10)

    def _on_msg(self, msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        now = time.monotonic()
        self.buffer.add(now, (p.x, p.y, self.yaw_sign * yaw))
        self.rate.tick(now)


@dataclass(frozen=True)
class Scan:
    stamp: float
    angles: np.ndarray    # rad, lidar frame, CCW from forward
    ranges: np.ndarray    # m, invalid returns removed


class ScanSubscriber:
    def __init__(self, topic: str):
        from unitree_sdk2py.core.channel import ChannelSubscriber

        from .laserscan_idl import LaserScan_

        self.rate = RateMeter()
        self._lock = threading.Lock()
        self._scan: Optional[Scan] = None
        self._sub = ChannelSubscriber(topic, LaserScan_)
        self._sub.Init(self._on_msg, 2)

    def _on_msg(self, msg) -> None:
        r = np.asarray(msg.ranges, dtype=np.float32)
        a = msg.angle_min + msg.angle_increment * np.arange(len(r), dtype=np.float32)
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        now = time.monotonic()
        with self._lock:
            self._scan = Scan(now, a[ok], r[ok])
        self.rate.tick(now)

    def latest(self) -> Optional[Scan]:
        with self._lock:
            return self._scan


def dds_twist_writer(topic: str) -> Callable[[float, float], None]:
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_, Vector3_

    pub = ChannelPublisher(topic, Twist_)
    pub.Init()

    def write(vx: float, wz: float) -> None:
        pub.Write(Twist_(linear=Vector3_(vx, 0.0, 0.0), angular=Vector3_(0.0, 0.0, wz)))

    return write


class CmdVelStreamer:
    """Publishes the latest (v, w) at a fixed rate while active.

    * deadman: a command older than cmd_timeout_s is replaced by zero, so a
      stalled control loop stops the base;
    * release(): publishes zero for zero_hold_s, then stops publishing so the
      handheld remote gets the channel back.
    """

    def __init__(self, write: Callable[[float, float], None], hz: float = 30.0, cmd_timeout_s: float = 0.3,
                 x_sign: float = 1.0, yaw_sign: float = 1.0):
        self._write = write
        self._period = 1.0 / hz
        self._timeout = cmd_timeout_s
        self._signs = (x_sign, yaw_sign)
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0)
        self._cmd_t = -math.inf
        self._active = False
        self._hold_until = -math.inf
        self._running = True
        self.sent = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def activate(self) -> None:
        with self._lock:
            self._active, self._hold_until, self._cmd = True, -math.inf, (0.0, 0.0)

    def set(self, v: float, w: float) -> None:
        with self._lock:
            self._cmd, self._cmd_t = (float(v), float(w)), time.monotonic()

    def release(self, zero_hold_s: float = 1.0, wait: bool = True) -> None:
        with self._lock:
            self._cmd, self._hold_until = (0.0, 0.0), time.monotonic() + zero_hold_s
        if wait:
            while self.active:
                time.sleep(self._period)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def current(self) -> tuple:
        """The command actually being published (after the deadman)."""
        with self._lock:
            return self._effective(time.monotonic())

    def _effective(self, now: float) -> tuple:
        if now > self._hold_until > -math.inf or now - self._cmd_t > self._timeout:
            return 0.0, 0.0
        return self._cmd

    def _loop(self) -> None:
        t_next = time.monotonic()
        while self._running:
            now = time.monotonic()
            with self._lock:
                publish = self._active
                if self._hold_until > -math.inf:
                    v, w = 0.0, 0.0
                    if now >= self._hold_until:
                        self._active, self._hold_until = False, -math.inf
                else:
                    v, w = self._effective(now)
            if publish:
                self._write(self._signs[0] * v, self._signs[1] * w)
                self.sent += 1
            t_next += self._period
            time.sleep(max(0.0, t_next - time.monotonic()))

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
