"""HTTP control API on the robot (stdlib only), for a web UI or a planner.

    GET  /health                 robot sensors, VLA reachability, episode state
    GET  /status                 live episode status (pose, v, w, replies, stops, result)
    POST /episode  {"instruction": "..."}    start an episode (409 if one is running)
    POST /stop     {"reason": "..."}         stop the running episode
    POST /estop/clear                        clear a latched chassis bumper / cliff e-stop (direct mode)
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("navgpt_g1d.api")


def serve(runner, robot, vla, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, payload, status=200):
            blob = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("access-control-allow-origin", "*")
            self.send_header("content-length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _body(self) -> dict:
            n = int(self.headers.get("content-length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}") if n else {}
            except ValueError:
                return {}

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/health":
                try:
                    vla_info, vla_ok = vla.info(), True
                except Exception as exc:  # noqa: BLE001
                    vla_info, vla_ok = repr(exc), False
                self._send({"robot": robot.health(), "vla_ok": vla_ok, "vla": vla_info,
                            "episode_running": runner.running})
            elif path == "/status":
                self._send(runner.status())
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            path, body = self.path.rstrip("/"), self._body()
            if path == "/episode":
                instruction = str(body.get("instruction") or "").strip()
                if not instruction:
                    self._send({"error": "instruction required"}, 400)
                elif runner.running:
                    self._send({"error": "an episode is already running"}, 409)
                else:
                    runner.start(instruction)
                    self._send({"started": True, "instruction": instruction})
            elif path == "/stop":
                runner.stop(str(body.get("reason") or "stop requested over API"))
                self._send({"stopping": runner.running})
            elif path == "/estop/clear":
                if runner.running:
                    self._send({"error": "stop the episode first"}, 409)
                else:
                    getattr(robot, "clear_estop", lambda: None)()
                    self._send({"cleared": True})
            else:
                self._send({"error": "not found"}, 404)

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print("[navgpt_g1d] control API on http://{}:{}".format(host, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop("API shutting down")
        runner.wait(10.0)
        httpd.server_close()
