"""SLAMWARE chassis HTTP events (bumper, cliff) polled in the background.

Endpoint and field names as used by g1d_ws (``/api/platform/v1/events``,
``type`` / ``timestamp``) on the chassis at 192.168.123.163:1448; check them in
the P0 interface step. Requests bypass any proxy.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request

from .supervisor import EventLatch

log = logging.getLogger("navgpt_g1d.chassis")
EVENTS = "/api/platform/v1/events"


def fetch_events(base_url: str, timeout_s: float = 0.5) -> list:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.rstrip("/") + EVENTS, timeout=timeout_s) as r:
        body = r.read()
    data = json.loads(body) if body else []
    if isinstance(data, dict):
        data = data.get("events", data.get("data", []))
    return [e for e in (data or []) if isinstance(e, dict)]


class ChassisEventPoller:
    def __init__(self, base_url: str, latch: EventLatch, period_s: float = 0.2):
        self.base_url, self.latch, self.period_s = base_url, latch, period_s
        self.ok = False
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for event in self.latch.update(fetch_events(self.base_url)):
                    log.warning("chassis event %s", event)
                self.ok = True
            except Exception as exc:  # noqa: BLE001 - keep polling; reported by `check`
                self.ok = False
                self.errors += 1
                if self.errors in (1, 50):
                    log.warning("chassis events unavailable at %s: %r", self.base_url, exc)
            self._stop.wait(self.period_s)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
