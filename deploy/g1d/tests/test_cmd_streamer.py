import threading
import time
import unittest

from navgpt_g1d.dds import CmdVelStreamer


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.sent = []

    def __call__(self, v, w):
        with self.lock:
            self.sent.append((time.monotonic(), v, w))

    def snapshot(self):
        with self.lock:
            return list(self.sent)


class CmdVelStreamerTest(unittest.TestCase):
    def setUp(self):
        self.rec = Recorder()
        self.s = CmdVelStreamer(self.rec, hz=50, cmd_timeout_s=0.1, x_sign=1.0, yaw_sign=-1.0)

    def tearDown(self):
        self.s.close()

    def test_silent_until_active(self):
        time.sleep(0.1)
        self.assertEqual(self.rec.snapshot(), [])

    def test_publishes_with_signs_and_deadman(self):
        self.s.activate()
        self.s.set(0.2, 0.3)
        time.sleep(0.06)
        last = self.rec.snapshot()[-1]
        self.assertEqual(last[1:], (0.2, -0.3))
        time.sleep(0.2)                                   # no new set(): deadman
        self.assertEqual(self.rec.snapshot()[-1][1:], (0.0, 0.0))

    def test_release_holds_zero_then_goes_quiet(self):
        self.s.activate()
        self.s.set(0.2, 0.0)
        time.sleep(0.05)
        t0 = time.monotonic()
        self.s.release(zero_hold_s=0.2)
        self.assertGreaterEqual(time.monotonic() - t0, 0.19)
        self.assertFalse(self.s.active)
        after = [r for r in self.rec.snapshot() if r[0] > t0]
        self.assertTrue(after and all(r[1:] == (0.0, 0.0) for r in after))
        n = len(self.rec.snapshot())
        time.sleep(0.1)
        self.assertEqual(len(self.rec.snapshot()), n, "channel released")


if __name__ == "__main__":
    unittest.main()
