"""Navigation episodes and the eight-waypoint targets NavGPT VLA is trained on.

An episode is one instruction plus an ordered list of frames; each frame holds
one image per camera and the odometry pose ``(x, y, yaw)`` at capture time.
Frames may be dense (teleoperation recorded at a few Hz) or already keyframed
(one per Habitat action, or the frames navgpt_g1d committed).

Keyframes follow the deployment rule (navgpt_g1d ``Navigator.next_request``): a
new keyframe once the robot moved ``keyframe_m`` or turned ``keyframe_rad``
since the previous one. Keyframes form the visual history. The same rule,
walked forward from the current pose, yields the targets: eight cumulative
poses in the current robot frame (+x forward, +y left, +theta CCW), padded
with the final pose (a stop) and divided by the embodiment scale.

Manifest format, one JSON object per line::

    {"episode_id": "g1d_0001",
     "instruction": "Walk to the meeting room door and stop.",   # or "instructions": [...]
     "frames": [{"image": "frames/00000.jpg", "pose": [x, y, yaw]}, ...]}

Multi-camera episodes use ``"images": [front, right, back, left]`` per frame.
Relative image paths are resolved against the manifest's directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

NUM_WAYPOINTS = 8
# The deployment's arrival test (navgpt_g1d.navigator.suggests_arrival).
ARRIVAL_RADIUS_M = 0.25
ARRIVAL_THETA_RAD = math.pi / 6
# Keyframe thresholds are compared with this slack so that exact Habitat
# actions (15 deg turns) pass a threshold written as 0.2618.
_THRESHOLD_SLACK = 1e-3

Pose = tuple


@dataclass
class Frame:
    images: tuple
    pose: Pose


@dataclass
class Episode:
    episode_id: str
    instructions: list
    frames: list
    meta: dict = field(default_factory=dict)

    @property
    def poses(self) -> np.ndarray:
        return np.asarray([f.pose for f in self.frames], dtype=np.float64)


def wrap_angle(a):
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def relative_pose(origin: Sequence[float], pose: Sequence[float]) -> tuple:
    """``pose`` expressed in the robot frame at ``origin``."""
    ox, oy, oyaw = origin
    dx, dy = pose[0] - ox, pose[1] - oy
    c, s = math.cos(oyaw), math.sin(oyaw)
    return c * dx + s * dy, -s * dx + c * dy, float(wrap_angle(pose[2] - oyaw))


def moved_enough(a: Sequence[float], b: Sequence[float], keyframe_m: float, keyframe_rad: float) -> bool:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    turn = abs(float(wrap_angle(b[2] - a[2])))
    return dist >= keyframe_m * (1 - _THRESHOLD_SLACK) or turn >= keyframe_rad * (1 - _THRESHOLD_SLACK)


def keyframe_indices(poses: np.ndarray, keyframe_m: float, keyframe_rad: float) -> list:
    """Indices of the frames the deployment would commit to the visual history."""
    keys = [0]
    for i in range(1, len(poses)):
        if moved_enough(poses[keys[-1]], poses[i], keyframe_m, keyframe_rad):
            keys.append(i)
    return keys


def future_waypoints(poses: np.ndarray, start: int, keyframe_m: float, keyframe_rad: float,
                     num: int = NUM_WAYPOINTS) -> np.ndarray:
    """Metric targets ``(num, 3)`` for the robot at frame ``start``.

    Each waypoint is the first later pose that moved or turned a keyframe step
    from the previous waypoint, like one Habitat action. The episode's final
    pose closes the trajectory and pads the rest, which is how a stop looks.
    """
    origin = poses[start]
    out, last = [], origin
    for i in range(start + 1, len(poses)):
        if len(out) == num:
            break
        if moved_enough(last, poses[i], keyframe_m, keyframe_rad):
            out.append(relative_pose(origin, poses[i]))
            last = poses[i]
    final = relative_pose(origin, poses[-1])
    if len(out) < num and (not out or not np.allclose(out[-1], final, atol=1e-4)):
        out.append(final)
    while len(out) < num:
        out.append(out[-1] if out else (0.0, 0.0, 0.0))
    return np.asarray(out[:num], dtype=np.float64)


def normalize(waypoints: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    return (np.asarray(waypoints, dtype=np.float64) / np.asarray(scale, dtype=np.float64)).reshape(-1)


def denormalize(flat: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(-1, 3) * np.asarray(scale, dtype=np.float64)


def is_arrival(waypoints: np.ndarray) -> bool:
    x, y, theta = waypoints[-1]
    return math.hypot(x, y) < ARRIVAL_RADIUS_M and abs(theta) < ARRIVAL_THETA_RAD


def motion_class(waypoints: np.ndarray, turn_rad: float = math.radians(30), lateral_m: float = 0.3) -> str:
    """``stop``, ``turn`` or ``straight``, used to rebalance the forward-heavy raw data."""
    if is_arrival(waypoints):
        return "stop"
    if np.max(np.abs(waypoints[:, 2])) >= turn_rad or abs(waypoints[-1, 1]) >= lateral_m:
        return "turn"
    return "straight"


def sample_history(n: int, size: int, rng: np.random.Generator, latest: bool = False) -> list:
    """History positions fed to the model, as in ``navgpt.vla.agent``.

    ``random`` keeps the first and last positions and samples the rest
    uniformly without replacement; ``latest`` keeps the last ``size``.
    """
    if n <= size:
        return list(range(n))
    if latest:
        return list(range(n - size, n))
    if size == 1:
        return [n - 1]
    middle = rng.choice(np.arange(1, n - 1), size=size - 2, replace=False)
    return sorted({0, n - 1, *(int(i) for i in middle)})


def load_episodes(manifest: str, view_num: int = 1) -> list:
    path = Path(manifest)
    root = path.parent
    episodes = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            where = "{}:{}".format(path, line_no)
            instructions = raw.get("instructions") or ([raw["instruction"]] if raw.get("instruction") else [])
            if not instructions:
                raise ValueError("{}: episode has no instruction".format(where))
            frames = []
            for f in raw["frames"]:
                images = f.get("images") or [f["image"]]
                if len(images) != view_num:
                    raise ValueError("{}: frame has {} image(s), view_num is {}".format(where, len(images), view_num))
                pose = tuple(float(v) for v in f["pose"])
                if len(pose) != 3:
                    raise ValueError("{}: pose must be [x, y, yaw]".format(where))
                frames.append(Frame(tuple(str(_resolve(root, p)) for p in images), pose))
            if len(frames) < 2:
                continue
            meta = {k: v for k, v in raw.items() if k not in ("episode_id", "instruction", "instructions", "frames")}
            episodes.append(Episode(str(raw.get("episode_id", where)), list(instructions), frames, meta))
    return episodes


def _resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


@dataclass
class NavIndex:
    """Every keyframe of every episode, with its keyframe-time target and class."""

    episodes: list
    keyframes: list            # per episode: frame indices of its keyframes
    items: list                # (episode index, keyframe position)
    targets: list              # metric (8, 3) target per item, current frame = keyframe
    classes: list

    @classmethod
    def build(cls, episodes: list, keyframe_m: float, keyframe_rad: float) -> "NavIndex":
        keyframes, items, targets, classes = [], [], [], []
        for e, ep in enumerate(episodes):
            poses = ep.poses
            keys = keyframe_indices(poses, keyframe_m, keyframe_rad)
            keyframes.append(keys)
            for j, k in enumerate(keys):
                wp = future_waypoints(poses, k, keyframe_m, keyframe_rad)
                items.append((e, j))
                targets.append(wp)
                classes.append(motion_class(wp))
        return cls(episodes, keyframes, items, targets, classes)

    def class_counts(self) -> dict:
        out = {"stop": 0, "turn": 0, "straight": 0}
        for c in self.classes:
            out[c] += 1
        return out


@dataclass
class NavSample:
    """One training example before tokenisation."""

    instruction: str
    history: list              # frames in time order; the last one is the current observation
    history_steps: list        # history positions used in the "Time step k" prompt labels
    waypoints: np.ndarray      # metric (8, 3)


def make_sample(index: NavIndex, item: int, rng: np.random.Generator, keyframe_m: float, keyframe_rad: float,
                max_sample_size: int, p_latest: float = 0.0, p_off_keyframe: float = 0.0,
                instruction: Optional[str] = None) -> NavSample:
    """History and target for one item.

    With probability ``p_off_keyframe`` the current observation is a frame
    between this keyframe and the next, as when the deployment predicts from
    an uncommitted frame; the committed keyframes stay the history before it.
    """
    e, j = index.items[item]
    ep, keys = index.episodes[e], index.keyframes[e]
    positions = list(keys[: j + 1])
    current = keys[j]
    stop = keys[j + 1] if j + 1 < len(keys) else len(ep.frames)
    between = list(range(keys[j] + 1, stop))
    if between and rng.random() < p_off_keyframe:
        current = int(rng.choice(between))
        positions.append(current)
        waypoints = future_waypoints(ep.poses, current, keyframe_m, keyframe_rad)
    else:
        waypoints = index.targets[item]
    steps = sample_history(len(positions), max_sample_size, rng, latest=rng.random() < p_latest)
    if instruction is None:
        instruction = ep.instructions[int(rng.integers(len(ep.instructions)))]
    return NavSample(instruction, [ep.frames[positions[s]] for s in steps], steps, waypoints)
