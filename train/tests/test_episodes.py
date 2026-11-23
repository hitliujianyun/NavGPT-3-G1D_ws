import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from navgpt_sft.episodes import (Episode, Frame, NavIndex, future_waypoints, keyframe_indices, load_episodes,
                                 make_sample, motion_class, normalize, relative_pose, sample_history)

KM, KR = 0.25, math.radians(15)


def line_poses(n, step, yaw=0.0):
    return np.asarray([[i * step * math.cos(yaw), i * step * math.sin(yaw), yaw] for i in range(n)])


def episode(poses):
    return Episode("e", ["go"], [Frame(("f{}.jpg".format(i),), tuple(p)) for i, p in enumerate(poses)])


class KeyframeTest(unittest.TestCase):
    def test_dense_straight_line(self):
        self.assertEqual(keyframe_indices(line_poses(21, 0.05), KM, KR), [0, 5, 10, 15, 20])

    def test_habitat_turns_pass_rounded_threshold(self):
        poses = np.asarray([[0, 0, i * math.radians(15)] for i in range(4)])
        self.assertEqual(keyframe_indices(poses, 0.25, 0.2618), [0, 1, 2, 3])


class WaypointTest(unittest.TestCase):
    def test_straight_targets_and_stop_padding(self):
        poses = line_poses(11, 0.25)
        wp = future_waypoints(poses, 0, KM, KR)
        np.testing.assert_allclose(wp[:, 0], [0.25 * k for k in range(1, 9)])
        np.testing.assert_allclose(wp[:, 1:], 0)
        tail = future_waypoints(poses, 8, KM, KR)
        np.testing.assert_allclose(tail[:, 0], [0.25, 0.5] + [0.5] * 6, err_msg="final pose pads the horizon")
        np.testing.assert_allclose(future_waypoints(poses, 10, KM, KR), 0, err_msg="last frame is a stop")

    def test_partial_last_step_is_kept(self):
        poses = np.asarray([[0, 0, 0], [0.25, 0, 0], [0.35, 0, 0]])
        np.testing.assert_allclose(future_waypoints(poses, 0, KM, KR)[:3, 0], [0.25, 0.35, 0.35])

    def test_targets_are_in_the_robot_frame(self):
        poses = line_poses(9, 0.25, yaw=math.pi / 2)  # driving along world +y
        wp = future_waypoints(poses, 0, KM, KR)
        np.testing.assert_allclose(wp[:, 0], [0.25 * k for k in range(1, 9)], atol=1e-9)
        np.testing.assert_allclose(wp[:, 1:], 0, atol=1e-9)
        x, y, t = relative_pose((1.0, 1.0, math.pi / 2), (0.0, 1.0, math.pi))
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0, msg="+y is left")
        self.assertAlmostEqual(t, math.pi / 2, msg="+theta is CCW")

    def test_turn_in_place(self):
        poses = np.asarray([[0, 0, i * math.radians(5)] for i in range(13)])  # 60 deg left in 5 deg frames
        wp = future_waypoints(poses, 0, KM, KR)
        np.testing.assert_allclose(np.degrees(wp[:4, 2]), [15, 30, 45, 60])
        np.testing.assert_allclose(np.degrees(wp[4:, 2]), 60)
        self.assertEqual(motion_class(wp), "turn")

    def test_normalisation(self):
        wp = np.tile([1.0, 0.433, 2.0944], (8, 1))
        np.testing.assert_allclose(normalize(wp, (1.0, 0.433, 2.0944)), 1.0)

    def test_classes(self):
        self.assertEqual(motion_class(np.zeros((8, 3))), "stop")
        self.assertEqual(motion_class(future_waypoints(line_poses(11, 0.25), 0, KM, KR)), "straight")


class HistoryTest(unittest.TestCase):
    def test_random_keeps_first_and_last(self):
        rng = np.random.default_rng(0)
        for n in (17, 40):
            idx = sample_history(n, 16, rng)
            self.assertEqual(len(idx), 16)
            self.assertEqual((idx[0], idx[-1]), (0, n - 1))
            self.assertEqual(idx, sorted(idx))
        self.assertEqual(sample_history(5, 16, rng), [0, 1, 2, 3, 4])
        self.assertEqual(sample_history(20, 4, rng, latest=True), [16, 17, 18, 19])

    def test_off_keyframe_current_frame(self):
        poses = line_poses(21, 0.05)                  # keyframes at 0, 5, 10, 15, 20
        index = NavIndex.build([episode(poses)], KM, KR)
        rng = np.random.default_rng(1)
        s = make_sample(index, 1, rng, KM, KR, 16, p_off_keyframe=1.0)
        names = [f.images[0] for f in s.history]
        self.assertEqual(names[:2], ["f0.jpg", "f5.jpg"])
        current = int(names[-1][1:-4])
        self.assertTrue(5 < current < 10)
        self.assertEqual(s.history_steps, [0, 1, 2])
        self.assertAlmostEqual(s.waypoints[0, 0], poses[current + 5, 0] - poses[current, 0])
        on_key = make_sample(index, 1, rng, KM, KR, 16, p_off_keyframe=0.0)
        self.assertEqual([f.images[0] for f in on_key.history], ["f0.jpg", "f5.jpg"])


class ManifestTest(unittest.TestCase):
    def test_load_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "eps.jsonl"
            rec = {"episode_id": "a", "instructions": ["x", "y"],
                   "frames": [{"image": "img/0.jpg", "pose": [0, 0, 0]}, {"image": "img/1.jpg", "pose": [0.3, 0, 0]}]}
            short = {"episode_id": "b", "instruction": "z", "frames": [{"image": "0.jpg", "pose": [0, 0, 0]}]}
            path.write_text(json.dumps(rec) + "\n\n" + json.dumps(short) + "\n", encoding="utf-8")
            eps = load_episodes(str(path))
            self.assertEqual(len(eps), 1, "single-frame episodes are dropped")
            self.assertEqual(Path(eps[0].frames[1].images[0]), Path(d) / "img" / "1.jpg")
            self.assertEqual(eps[0].instructions, ["x", "y"])
            with self.assertRaises(ValueError):
                load_episodes(str(path), view_num=4)


if __name__ == "__main__":
    unittest.main()
