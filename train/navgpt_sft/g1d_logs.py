"""Episodes from navgpt_g1d run logs (``logs/<timestamp>/events.jsonl`` + ``frames/``).

Each committed request saved ``frames/<seq>.jpg`` (the 400x400 frame the VLA
saw) and its reply event holds ``capture_pose``; together they are the
episode's keyframes. These runs are the model's own behaviour, so only keep
those a person judged correct: by default runs that ended ``arrived``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def read_run(run_dir: Path) -> dict:
    instruction, status, frames = None, None, {}
    with (run_dir / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            ev = json.loads(line)
            kind = ev.get("event")
            if kind == "start":
                instruction = ev.get("instruction")
            elif kind == "result":
                status = ev.get("status")
            elif kind == "reply" and ev.get("commit") and ev.get("capture_pose") is not None:
                frames.setdefault(int(ev["seq"]), [float(v) for v in ev["capture_pose"]][:3])
    return {"instruction": instruction, "status": status, "frames": frames}


def convert(run_dirs: list, manifest: str, statuses=("arrived",), instruction: str = None) -> dict:
    """Write one manifest line per usable run; returns counts per outcome."""
    out_path = Path(manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "skipped_status": 0, "skipped_frames": 0, "skipped_instruction": 0}
    with out_path.open("w", encoding="utf-8") as out:
        for run_dir in sorted(Path(d) for d in run_dirs):
            if not (run_dir / "events.jsonl").exists():
                continue
            run = read_run(run_dir)
            if statuses and run["status"] not in statuses:
                counts["skipped_status"] += 1
                continue
            text = instruction or run["instruction"]
            if not text:
                counts["skipped_instruction"] += 1
                continue
            frames = []
            for seq in sorted(run["frames"]):
                image = run_dir / "frames" / "{:05d}.jpg".format(seq)
                if image.exists():
                    frames.append({"image": _relative(image, out_path.parent), "pose": run["frames"][seq]})
            if len(frames) < 2:
                counts["skipped_frames"] += 1
                continue
            record = {"episode_id": run_dir.name, "instruction": text, "frames": frames,
                      "source": "g1d_log", "status": run["status"]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["written"] += 1
    return counts


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()
    except ValueError:  # different drives on Windows
        return path.resolve().as_posix()
