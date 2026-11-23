"""Motion authority and the final command filter (NavGPT-3 paper, Appendix B.4).

At most one operation holds the permission to move. Every grant or revoke bumps
a version number, and a command is forwarded only under the current version, so
late model outputs or queued commands cannot regain authority after a revoke. A
protective stop is latched: clearing it does not resume motion, a fresh grant
is required.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Grant:
    operation: str
    version: int


class MotionAuthority:
    def __init__(self):
        self._lock = threading.Lock()
        self._version = 0
        self._holder: Optional[str] = None
        self._protective_stop = False
        self._stop_reason = ""

    def grant(self, operation: str) -> Optional[Grant]:
        with self._lock:
            if self._protective_stop:
                return None
            self._version += 1
            self._holder = operation
            return Grant(operation, self._version)

    def revoke(self, reason: str = "") -> None:
        with self._lock:
            self._version += 1
            self._holder = None
            self._stop_reason = reason

    def protective_stop(self, reason: str) -> None:
        """Highest-priority event: revoke at once, without waiting for any model."""
        with self._lock:
            self._protective_stop = True
            self._version += 1
            self._holder = None
            self._stop_reason = reason

    def clear_protective_stop(self) -> None:
        with self._lock:
            self._protective_stop = False

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._protective_stop

    @property
    def stop_reason(self) -> str:
        with self._lock:
            return self._stop_reason

    def holds(self, grant: Optional[Grant]) -> bool:
        with self._lock:
            return (grant is not None and not self._protective_stop
                    and self._holder == grant.operation and grant.version == self._version)

    def filter(self, v: float, w: float, grant: Optional[Grant]) -> Tuple[float, float]:
        return (v, w) if self.holds(grant) else (0.0, 0.0)
