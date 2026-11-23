"""Smoothness and latency metrics from an episode's events.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def analyse(log_dir) -> dict:
    events = [json.loads(line) for line in Path(log_dir, "events.jsonl").read_text(encoding="utf-8").splitlines()]
    cmds = [e for e in events if e["event"] == "cmd"]
    replies = [e for e in events if e["event"] == "reply"]
    result = next((e for e in events if e["event"] == "result"), {})
    out = {"status": result.get("status"), "reason": result.get("reason"),
           "duration_s": result.get("duration_s"), "distance_m": result.get("distance_m"),
           "protective_stops": sum(e["event"] == "protective_stop" for e in events),
           "act_failures": sum(e["event"] == "act_failed" for e in events)}
    if len(cmds) > 2:
        t = np.array([e["t"] for e in cmds])
        v = np.array([e["v"] for e in cmds])
        w = np.array([e["w"] for e in cmds])
        phase = np.array([e.get("phase", "running") for e in cmds])
        running = phase == "running"
        out["phase_s"] = {p: round(float(np.sum(phase == p)) / max(len(t) / (t[-1] - t[0]), 1e-6), 1)
                          for p in sorted(set(phase.tolist()))}
        moving = np.nonzero(v > 0.05)[0]
        if len(moving):
            span = slice(moving[0], moving[-1] + 1)
            out["mean_speed_mps"] = round(float(v[span].mean()), 3)
            free = running[span]
            out["stall_ratio"] = round(float(np.mean(v[span][free] < 0.03)) if free.any() else 0.0, 3)
        dt = np.array([e["dt"] for e in cmds[1:]]) if "dt" in cmds[0] else np.diff(t)
        dt = np.maximum(dt, 1e-3)
        smooth = running[1:] & running[:-1]       # protective stops are deliberately abrupt
        if smooth.any():
            out["max_accel_mps2"] = round(float(np.max(np.abs(np.diff(v))[smooth] / dt[smooth])), 3)
            out["max_ang_accel_rps2"] = round(float(np.max(np.abs(np.diff(w))[smooth] / dt[smooth])), 3)
        out["control_hz"] = round(float(1.0 / np.median(np.maximum(np.diff(t), 1e-3))), 1)
    if replies:
        lat = np.array([e["latency_s"] for e in replies])
        out["replies"] = len(replies)
        out["accepted_ratio"] = round(float(np.mean([e["accepted"] for e in replies])), 3)
        out["committed_frames"] = int(sum(e["commit"] for e in replies))
        out["latency_p50_s"] = round(float(np.percentile(lat, 50)), 3)
        out["latency_p95_s"] = round(float(np.percentile(lat, 95)), 3)
        infer = [e["infer_s"] for e in replies if e.get("infer_s") is not None]
        if infer:
            out["infer_p50_s"] = round(float(np.percentile(infer, 50)), 3)
        dev = [e["deviation_m"] for e in replies if e.get("deviation_m") is not None]
        if dev:
            out["splice_deviation_p95_m"] = round(float(np.percentile(dev, 95)), 3)
    return out
