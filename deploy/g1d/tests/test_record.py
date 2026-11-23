import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from navgpt_g1d.pose_buffer import PoseBuffer
from navgpt_g1d.record import EpisodeRecorder

TRAIN = Path(__file__).resolve().parents[3] / "train"


class FakeRobot:
    """Straight drive at 0.4 m/s with a left turn; frames at 30 Hz, odometry at 20 Hz."""

    def __init__(self):
        self.odom = PoseBuffer()
        self.t0 = time.monotonic()

    def pose(self, t):
        s = 0.4 * t
        return (s, 0.0, 0.0) if s < 1.0 else (1.0, s - 1.0, math.pi / 2)

    def frame(self):
        now = time.monotonic()
        t = now - self.t0
        for k in range(int(t * 20) - 2, int(t * 20) + 1):
            if k >= 0:
                self.odom.add(self.t0 + k / 20.0, self.pose(k / 20.0))
        rgb = np.full((400, 400, 3), int(t * 50) % 255, dtype=np.uint8)
        return rgb, now


class RecordTest(unittest.TestCase):
    def test_episode_round_trip_into_navgpt_sft(self):
        with tempfile.TemporaryDirectory() as d:
            rec = EpisodeRecorder(FakeRobot(), d, latency_s=0.05, rate_hz=20.0, episode_id="ep1")
            rec.run(lambda: False, max_s=5.0)
            self.assertGreater(len(rec.frames), 60)
            line = rec.finish(["Go forward, then turn left.", " "], "train.jsonl")
            self.assertEqual(line["instructions"], ["Go forward, then turn left."])
            stored = json.loads(Path(d, "train.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(Path(d, stored["frames"][0]["image"]).exists())
            ys = [f["pose"][1] for f in stored["frames"]]
            self.assertTrue(all(b >= a - 1e-9 for a, b in zip(ys, ys[1:])), "poses follow the drive")

            if not (TRAIN / "navgpt_sft" / "episodes.py").exists():
                self.skipTest("train/ not present (robot package): manifest checked, loader not")
            sys.path.insert(0, str(TRAIN))
            try:
                from navgpt_sft.episodes import NavIndex, load_episodes
            finally:
                sys.path.remove(str(TRAIN))
            episodes = load_episodes(str(Path(d, "train.jsonl")))
            index = NavIndex.build(episodes, 0.25, math.radians(15))
            self.assertEqual(len(episodes), 1)
            self.assertGreater(len(index.items), 5, "keyframes every 0.25 m over ~2 m")
            self.assertIn("turn", index.class_counts())

    def test_short_or_unlabelled_episode_is_not_written(self):
        with tempfile.TemporaryDirectory() as d:
            rec = EpisodeRecorder(FakeRobot(), d, latency_s=0.05, rate_hz=20.0)
            self.assertIsNone(rec.finish(["x"], "train.jsonl"))
            rec.run(lambda: False, max_s=0.5)
            self.assertIsNone(rec.finish([""], "train.jsonl"))
            self.assertFalse(Path(d, "train.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
