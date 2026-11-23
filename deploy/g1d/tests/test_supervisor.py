import json
import unittest

from navgpt_g1d.supervisor import CLEAR, EventLatch, GuardState, JoystickMonitor, Supervision, combine


class GuardStateTest(unittest.TestCase):
    def state(self, now=10.0, **fields):
        g = GuardState(timeout_s=0.5)
        base = {"blocked": "", "estop": False, "manual": False}
        g.update(json.dumps({**base, **fields}), now=now)
        return g

    def test_missing_and_stale_state_is_lost(self):
        self.assertEqual(GuardState().verdict(1.0).abort_status, "guard_lost")
        self.assertEqual(self.state(now=10.0).verdict(10.6).abort_status, "guard_lost")

    def test_estop_and_manual_abort(self):
        self.assertEqual(self.state(estop=True, estop_reason="bumper").verdict(10.1).abort_status, "estop")
        v = self.state(manual=True, manual_source="remote").verdict(10.1)
        self.assertEqual(v.abort_status, "manual_override")
        self.assertIn("remote", v.abort_reason)

    def test_blocked_does_not_abort(self):
        v = self.state(blocked="front").verdict(10.1)
        self.assertFalse(v.abort)
        self.assertEqual(v.blocked, "front")
        self.assertEqual(self.state().verdict(10.1), CLEAR)

    def test_garbage_is_ignored(self):
        g = self.state()
        g.update("not json", now=11.0)
        self.assertEqual(g.verdict(10.2), CLEAR)


class JoystickTest(unittest.TestCase):
    def test_deadzone_and_start_time(self):
        j = JoystickMonitor(deadzone=0.15)
        j.update(0.1, 0.0, -0.1, 0.05, now=1.0)
        self.assertFalse(j.active_since(0.0))
        j.update(0.0, 0.4, 0.0, 0.0, now=2.0)
        self.assertTrue(j.active_since(1.5))
        self.assertFalse(j.active_since(2.5), "stick use before the episode started does not count")


class EventLatchTest(unittest.TestCase):
    def test_only_new_stop_events_latch(self):
        latch = EventLatch()
        old = [{"type": "BUMPER_TRIGGERED", "timestamp": 1}]
        self.assertEqual(latch.update(old), [])
        self.assertEqual(latch.latched, "", "events from before start-up never latch")
        latch.update(old + [{"type": "BATTERY_LOW", "timestamp": 2}])
        self.assertEqual(latch.latched, "")
        latch.update(old + [{"type": "CLIFF_DETECTED", "timestamp": 3}])
        self.assertEqual(latch.latched, "CLIFF_DETECTED")
        latch.clear()
        self.assertEqual(latch.latched, "")


class CombineTest(unittest.TestCase):
    def test_abort_beats_blocked(self):
        v = combine(Supervision(blocked="front"), Supervision("estop", "x"))
        self.assertEqual(v.abort_status, "estop")
        self.assertEqual(combine(Supervision(blocked="rear")).blocked, "rear")
        self.assertEqual(combine(), CLEAR)


if __name__ == "__main__":
    unittest.main()
