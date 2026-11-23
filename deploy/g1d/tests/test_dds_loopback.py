"""Real CycloneDDS loopback (unitree_sdk2py) for the DDS pieces: writer discovery,
guard state, remote sticks and the command streamer. Skipped without cyclonedds.

Uses DDS domain 37 so it never talks to a robot on the same network.
"""

import importlib.util
import json
import sys
import threading
import time
import unittest
from pathlib import Path

SDK = Path(__file__).resolve().parents[3] / "code" / "unitree_sdk2_python"
if importlib.util.find_spec("unitree_sdk2py") is None and SDK.exists():
    sys.path.insert(0, str(SDK))

HAVE_DDS = importlib.util.find_spec("cyclonedds") is not None and importlib.util.find_spec("unitree_sdk2py") is not None
DOMAIN = 37


def wait_for(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


@unittest.skipUnless(HAVE_DDS, "needs cyclonedds and unitree_sdk2py")
class DdsLoopbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from navgpt_g1d.dds import init_dds

        init_dds(DOMAIN)

    def test_writer_discovery(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_

        from navgpt_g1d.dds import find_writers

        topic = "rt/navgpt_test/cmd_vel"
        self.assertEqual(find_writers(topic, DOMAIN, 0.5), [])
        pub = ChannelPublisher(topic, Twist_)
        pub.Init()
        self.assertEqual(len(find_writers(topic, DOMAIN, 1.0)), 1)

    def test_guard_state_and_joystick(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

        from navgpt_g1d.dds import GuardStateSubscriber, JoystickSubscriber
        from navgpt_g1d.supervisor import GuardState, JoystickMonitor

        state, joy = GuardState(), JoystickMonitor(0.15)
        GuardStateSubscriber(state, "rt/navgpt_test/guard_state")
        JoystickSubscriber(joy, "rt/navgpt_test/wirelesscontroller")
        gpub = ChannelPublisher("rt/navgpt_test/guard_state", String_)
        gpub.Init()
        jpub = ChannelPublisher("rt/navgpt_test/wirelesscontroller", WirelessController_)
        jpub.Init()
        t0 = time.monotonic()

        def send():
            for _ in range(40):
                gpub.Write(String_(data=json.dumps({"blocked": "front", "estop": False, "manual": False})))
                jpub.Write(WirelessController_(lx=0.0, ly=0.6, rx=0.0, ry=0.0, keys=0))
                time.sleep(0.05)

        threading.Thread(target=send, daemon=True).start()
        self.assertTrue(wait_for(lambda: state.verdict(time.monotonic()).blocked == "front"))
        self.assertTrue(wait_for(lambda: joy.active_since(t0)))

    def test_cmd_streamer_over_dds(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_

        from navgpt_g1d.dds import CmdVelStreamer, dds_twist_writer

        got = []
        sub = ChannelSubscriber("rt/navgpt_test/stream", Twist_)
        sub.Init(lambda m: got.append((m.linear.x, m.angular.z)), 10)
        streamer = CmdVelStreamer(dds_twist_writer("rt/navgpt_test/stream"), hz=30, cmd_timeout_s=0.3, yaw_sign=-1.0)
        try:
            streamer.activate()
            for _ in range(10):
                streamer.set(0.25, 0.4)
                time.sleep(0.03)
            self.assertTrue(wait_for(lambda: (0.25, -0.4) in [(round(a, 3), round(b, 3)) for a, b in got]))
            streamer.release(zero_hold_s=0.1)
            self.assertEqual(got[-1], (0.0, 0.0))
        finally:
            streamer.close()


@unittest.skipUnless(HAVE_DDS, "needs cyclonedds and unitree_sdk2py")
class RobotWiringTest(unittest.TestCase):
    """G1DRobot preflight and output routing, against DDS stand-ins for the robot."""

    @classmethod
    def setUpClass(cls):
        from navgpt_g1d.dds import init_dds

        init_dds(DOMAIN)

    def cfg(self, tag: str, **base):
        import dataclasses

        from navgpt_g1d.config import load_config

        cfg = load_config(None)
        topics = {k: "rt/navgpt_wiring_{}/{}".format(tag, k) for k in
                  ("odom", "scan", "cmd_vel", "guard_cmd", "guard_state", "joystick")}
        return dataclasses.replace(
            cfg, dds=dataclasses.replace(cfg.dds, domain_id=DOMAIN),
            topics=dataclasses.replace(cfg.topics, **topics),
            base=dataclasses.replace(cfg.base, **base),
            safety=dataclasses.replace(cfg.safety, enabled=False),
            supervisor=dataclasses.replace(cfg.supervisor, chassis_http=None))

    def test_direct_mode_refuses_a_second_writer(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_

        from navgpt_g1d.robot import G1DRobot, PreflightError

        cfg = self.cfg("direct")
        other = ChannelPublisher(cfg.topics.cmd_vel, Twist_)
        other.Init()
        with self.assertRaises(PreflightError):
            G1DRobot(cfg, with_camera=False)

    def test_guard_mode_needs_the_guard_then_routes_commands_to_it(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        from navgpt_g1d.robot import G1DRobot, PreflightError

        cfg = self.cfg("guard", output="guard")
        t_start = time.monotonic()
        with self.assertRaises(PreflightError):
            G1DRobot(cfg, with_camera=False)
        self.assertGreater(time.monotonic() - t_start, 2.5, "waits for the guard before giving up")

        state_pub = ChannelPublisher(cfg.topics.guard_state, String_)
        state_pub.Init()
        running = threading.Event()
        running.set()

        def guard():
            while running.is_set():
                state_pub.Write(String_(data=json.dumps({"blocked": "", "estop": False, "manual": False})))
                time.sleep(0.05)

        threading.Thread(target=guard, daemon=True).start()
        to_guard, to_base = [], []
        ChannelSubscriber(cfg.topics.guard_cmd, Twist_).Init(lambda m: to_guard.append(m.linear.x), 10)
        ChannelSubscriber(cfg.topics.cmd_vel, Twist_).Init(lambda m: to_base.append(m.linear.x), 10)
        robot = G1DRobot(cfg, with_camera=False)
        try:
            self.assertFalse(robot.supervise(time.monotonic(), 0.0).abort)
            robot.cmd.activate()
            for _ in range(10):
                robot.cmd.set(0.2, 0.0)
                time.sleep(0.03)
            self.assertTrue(wait_for(lambda: 0.2 in [round(v, 3) for v in to_guard]))
            self.assertEqual(to_base, [], "guard mode never writes cmd_vel_no_limit")
            running.clear()
            self.assertTrue(wait_for(lambda: robot.supervise(time.monotonic(), 0.0).abort_status == "guard_lost"))
        finally:
            running.clear()
            robot.cmd.release(0.0)
            robot.close()


if __name__ == "__main__":
    unittest.main()
