"""HTTP client for the NavGPT VLA service (server/serve_vla.py on the GPU host).

Replies from serve_vla.py carry ``waypoints``; an existing service that answers
``{"navigation": [...]}`` or the upstream ``[action, navigation]`` array also
works, but ignores ``commit`` and keeps every frame in its history.
"""

from __future__ import annotations

import base64
from typing import Optional

import numpy as np


class VlaClient:
    def __init__(self, url: str, timeout_s: float = 4.0, jpeg_quality: int = 85):
        import requests

        self._session = requests.Session()
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.jpeg_quality = jpeg_quality

    def info(self) -> dict:
        r = self._session.get(self.url + "/info", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def reset(self, seed: Optional[int] = None) -> None:
        r = self._session.post(self.url + "/reset", json={"seed": seed}, timeout=self.timeout_s)
        r.raise_for_status()

    def act(self, rgb: np.ndarray, instruction: str, commit: bool = True, seq: Optional[int] = None) -> dict:
        r = self._session.post(self.url + "/act",
                               json={"rgb_front": self._encode(rgb), "text": instruction,
                                     "commit": commit, "seq": seq, "is_stuck": False},
                               timeout=self.timeout_s)
        r.raise_for_status()
        return normalise_reply(r.json())

    def _encode(self, rgb: np.ndarray) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")


def normalise_reply(body) -> dict:
    if isinstance(body, list):
        body = {"navigation": body[1] if len(body) > 1 else []}
    if "error" in body:
        raise RuntimeError("VLA error: {}".format(body["error"]))
    waypoints = body.get("waypoints") or body.get("navigation")
    if not waypoints or len(waypoints) < 2:
        raise RuntimeError("VLA reply without waypoints: {}".format(str(body)[:200]))
    return {**body, "waypoints": [[float(v) for v in wp[:3]] for wp in waypoints]}
