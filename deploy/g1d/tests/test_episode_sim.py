"""End-to-end: EpisodeRunner with all its threads against the simulated base (real time)."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from navgpt_g1d.config import load_config
from navgpt_g1d.episode import EpisodeRunner
from navgpt_g1d.report import analyse
from navgpt_g1d.sim import Obstacle, RouteOracleVLA, SimRobot, SimWorld

ROUTE = [(0.0, 0.0), (1.6, 0.0), (1.6, 1.2)]


def run(obstacles=(), seed=0, v_max=0.4):
    cfg = load_config(None)
    tmp = tempfile.mkdtemp()
    tracker = dataclasses.replace(cfg.tracker, v_max=v_max)
    cfg = dataclasses.replace(cfg, tracker=tracker,
                              navigator=dataclasses.replace(cfg.navigator, tracker=tracker),
                              episode=dataclasses.replace(cfg.episode, log_dir=tmp, save_frames=False,
                                                          max_time_s=40.0))
    world = SimWorld(obstacles=obstacles, odom_noise_m=0.005, seed=seed)
    robot = SimRobot(world, cfg.base.publish_hz, cfg.base.cmd_timeout_s)
    vla = RouteOracleVLA(world, ROUTE, latency_s=(0.2, 0.6), stall_prob=0.0, seed=seed)
    try:
        result = EpisodeRunner(cfg, robot, vla).run("sim")
        end = world.pose
    finally:
        robot.close()
    return result, end, analyse(result.log_dir)


class EpisodeSimTest(unittest.TestCase):
    def test_smooth_arrival(self):
        result, end, report = run()
        self.assertEqual(result.status, "arrived", result.reason)
        self.assertLess(np.hypot(end[0] - 1.6, end[1] - 1.2), 0.35)
        self.assertLess(report["stall_ratio"], 0.03)
        self.assertLessEqual(report["max_accel_mps2"], 0.6 + 1e-3)
        self.assertLessEqual(report["max_ang_accel_rps2"], 1.2 + 1e-3)
        self.assertGreater(report["mean_speed_mps"], 0.22)
        self.assertGreater(report["committed_frames"], 5)
        self.assertLess(report["committed_frames"], report["replies"], "not every frame enters the history")

    def test_person_crossing_stops_and_resumes(self):
        crossing = Obstacle(x=1.1, y=-1.0, vy=0.5, radius=0.2, t_on=1.0)
        result, end, report = run(obstacles=[crossing])
        self.assertEqual(result.status, "arrived", result.reason)
        events = [json.loads(line)["event"]
                  for line in Path(result.log_dir, "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIn("protective_stop", events)
        self.assertIn("resume", events)
        self.assertGreater(events.index("resume"), events.index("protective_stop"))


if __name__ == "__main__":
    unittest.main()
