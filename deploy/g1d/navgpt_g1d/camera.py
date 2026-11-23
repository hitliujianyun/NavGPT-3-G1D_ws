"""Head-camera frames from Unitree teleimager, shaped for NavGPT VLA.

teleimager publishes each camera as raw JPEG bytes on a ZMQ PUB socket
(head camera: port 55555 by default, binocular frames side by side). The VLA
was trained on 400x400 square images at 120 deg HFOV, so frames are reduced to
one eye, centre-cropped to a square and resized. Frames are stamped with the
local monotonic clock on receipt.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np


class HeadCamera:
    def __init__(self, host: str = "192.168.123.164", port: int = 55555,
                 binocular: bool = True, eye: str = "left", out_size: int = 400,
                 stale_after_s: float = 0.3):
        import zmq

        self.binocular, self.eye, self.out_size = binocular, eye, out_size
        self.stale_after_s = stale_after_s

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect("tcp://{}:{}".format(host, port))
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self._lock = threading.Lock()
        self._jpg: Optional[bytes] = None
        self._stamp = 0.0
        self._times: deque = deque(maxlen=30)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while self._running:
            if self._sock in dict(poller.poll(timeout=100)):
                data = self._sock.recv()
                now = time.monotonic()
                with self._lock:
                    self._jpg, self._stamp = data, now
                    self._times.append(now)

    @property
    def rate_hz(self) -> float:
        with self._lock:
            ts = list(self._times)
        if len(ts) < 2 or time.monotonic() - ts[-1] > 1.0:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def latest(self) -> Optional[Tuple[np.ndarray, float]]:
        """(RGB uint8 HxWx3, receipt time) or None when no fresh frame exists."""
        import cv2

        with self._lock:
            jpg, stamp = self._jpg, self._stamp
        if jpg is None or time.monotonic() - stamp > self.stale_after_s:
            return None
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return preprocess(bgr[:, :, ::-1], self.binocular, self.eye, self.out_size), stamp

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._sock.close()


def preprocess(rgb: np.ndarray, binocular: bool, eye: str, out_size: int) -> np.ndarray:
    import cv2

    if binocular:
        half = rgb.shape[1] // 2
        rgb = rgb[:, :half] if eye == "left" else rgb[:, half:]
    h, w = rgb.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    square = rgb[top:top + side, left:left + side]
    return np.ascontiguousarray(cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_AREA))
