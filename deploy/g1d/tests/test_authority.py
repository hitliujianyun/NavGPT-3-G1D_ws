import unittest

from navgpt_g1d.authority import MotionAuthority

GO = (0.2, 0.1)
ZERO = (0.0, 0.0)


class AuthorityTest(unittest.TestCase):
    def test_holder_passes(self):
        auth = MotionAuthority()
        self.assertEqual(auth.filter(*GO, auth.grant("vla")), GO)

    def test_no_grant_blocks(self):
        self.assertEqual(MotionAuthority().filter(*GO, None), ZERO)

    def test_stale_grant_blocked_after_handoff(self):
        auth = MotionAuthority()
        old = auth.grant("vla")
        new = auth.grant("repair")
        self.assertEqual(auth.filter(*GO, old), ZERO)
        self.assertEqual(auth.filter(*GO, new), GO)

    def test_protective_stop_latches(self):
        auth = MotionAuthority()
        grant = auth.grant("vla")
        auth.protective_stop("lidar")
        self.assertEqual(auth.filter(*GO, grant), ZERO)
        self.assertIsNone(auth.grant("vla"))
        auth.clear_protective_stop()
        self.assertEqual(auth.filter(*GO, grant), ZERO, "clearing must not revive the old grant")
        self.assertEqual(auth.filter(*GO, auth.grant("vla")), GO)


if __name__ == "__main__":
    unittest.main()
