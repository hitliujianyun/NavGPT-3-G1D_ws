"""Single-view NavGPT VLA service for the G1-D head camera (runs on the GPU host).

The upstream service (code/NavGPT-3/navgpt/vla/api.py) is fixed to four Habitat
views and returns a TELEPORT action. This one keeps a front-only history (the
configuration the paper used on the Unitree Go2) and returns the eight metric
waypoints; the robot side anchors, splices and tracks them.

Run it inside the NavGPT-3 VLA environment:

    code/NavGPT-3/.envs/vla/bin/python deploy/g1d/server/serve_vla.py \
        --checkpoint models/NavGPT3-8B --port 8100 --gpu 0

Endpoints:
    GET  /info, /health
    POST /reset    {"seed": int | null}
    POST /act      {"rgb_front": <base64 JPEG/PNG>, "text": <instruction>,
                    "commit": bool = true, "seq": any = null}
                   -> {"waypoints": 8 x [x, y, theta], "arrival": bool, "infer_s": float,
                       "seq": <echoed>, "frames": <history length>}
                   commit=false predicts with the frame as the current observation
                   but does not keep it in the visual history.
    POST /observe  {"rgb_front": ...}  append a frame without predicting

The service is stateful (one episode at a time) and serialises all requests.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import threading
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "code" / "NavGPT-3"))

log = logging.getLogger("serve_vla")


def decode(b64: str):
    import cv2
    import numpy as np

    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def act_without_commit(agent, image, text):
    """Predict with ``image`` as the current frame, then drop it from the history."""
    from navgpt.vla.agent import recover_absolute_waypoints

    agent.observe_history([image])
    step = len(agent.rgb_lists[0]) - 1
    try:
        return recover_absolute_waypoints(agent.predict_inference(text), agent._scale)
    finally:
        agent.rgb_lists[0].pop()
        agent.multi_res_cache.pop(step, None)


def suggests_arrival(waypoints) -> bool:
    x, y, theta = waypoints[-1]
    return (x * x + y * y) ** 0.5 < 0.25 and abs(theta) < 3.141592653589793 / 6


def warm_up(agent, steps: int = 3) -> None:
    """Run a few dummy inferences so CUDA kernels and caches are ready, then reset."""
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(steps):
        t0 = time.time()
        agent.act([rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)], "Walk forward and stop.")
        log.info("warm-up inference %.3f s", time.time() - t0)
    agent.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--visual-token-budget", type=int, default=3072)
    parser.add_argument("--max-sample-size", type=int, default=16)
    parser.add_argument("--sample-mode", choices=["random", "latest"], default="random")
    parser.add_argument("--frame-cache", choices=["precompute", "on_demand"], default="precompute")
    parser.add_argument("--attn-implementation", default=None, help="sdpa | flash_attention_2 (default: auto)")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from flask import Flask, jsonify, request
    from navgpt.vla.agent import NavGPTVLA

    agent = NavGPTVLA(
        model_path=args.checkpoint,
        view_num=1,
        max_nav_vis_tokens=args.visual_token_budget,
        max_sample_size=args.max_sample_size,
        sample_mode=args.sample_mode,
        frame_cache=args.frame_cache,
        robot_type="habitat_nav",
        attn_implementation=args.attn_implementation,
    )
    if not args.no_warmup:
        warm_up(agent)
    lock = threading.Lock()
    app = Flask(__name__)

    @app.get("/info")
    @app.get("/health")
    def info():
        return jsonify({"model": os.path.basename(args.checkpoint.rstrip("/\\")), "view_num": 1,
                        "visual_token_budget": args.visual_token_budget,
                        "max_sample_size": args.max_sample_size, "frames": len(agent.rgb_lists[0])})

    @app.post("/reset")
    def reset():
        data = request.get_json(silent=True) or {}
        with lock:
            agent.reset()
            if data.get("seed") is not None:
                import random

                import numpy as np
                import torch

                seed = int(data["seed"])
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
        return jsonify({"status": "reset"})

    @app.post("/observe")
    def observe():
        data = request.get_json()
        with lock:
            agent.observe_history([decode(data["rgb_front"])])
            return jsonify({"frames": len(agent.rgb_lists[0])})

    @app.post("/act")
    def act():
        try:
            data = request.get_json()
            image = decode(data["rgb_front"])
            with lock:
                t0 = time.time()
                if data.get("commit", True):
                    waypoints = agent.act([image], data["text"])
                else:
                    waypoints = act_without_commit(agent, image, data["text"])
                infer_s = time.time() - t0
                frames = len(agent.rgb_lists[0])
        except Exception as exc:  # noqa: BLE001 - report to the robot, keep serving
            log.exception("/act failed")
            return jsonify({"error": repr(exc)}), 500
        waypoints = [[float(v) for v in wp] for wp in waypoints]
        return jsonify({"waypoints": waypoints, "arrival": suggests_arrival(waypoints), "infer_s": infer_s,
                        "seq": data.get("seq"), "frames": frames})

    log.info("serving %s on %s:%d", args.checkpoint, args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
