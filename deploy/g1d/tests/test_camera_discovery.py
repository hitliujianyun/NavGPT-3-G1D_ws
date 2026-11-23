import importlib.util
import threading
import unittest

from navgpt_g1d.camera import camera_entry, resolve_camera

NEW = {"camera": {"head_camera": {"color_zmq_port": 55555, "binocular": True, "image_shape": [480, 1280]},
                  "left_wrist_camera": {"color_zmq_port": 55556, "binocular": False}}}
OLD = {"head_camera": {"zmq_port": 55560, "image_shape": [480, 1280]}}


class CameraEntryTest(unittest.TestCase):
    def test_current_teleimager_schema(self):
        self.assertEqual(camera_entry(NEW), (55555, True))
        self.assertEqual(camera_entry(NEW, "left_wrist_camera"), (55556, False))

    def test_older_flat_schema_infers_binocular_from_shape(self):
        self.assertEqual(camera_entry(OLD), (55560, True))

    def test_missing_camera(self):
        with self.assertRaises(ValueError):
            camera_entry({"camera": {}}, "head_camera")

    def test_configured_values_skip_the_query(self):
        self.assertEqual(resolve_camera("unreachable.invalid", 55555, False), (55555, False))


@unittest.skipIf(importlib.util.find_spec("zmq") is None, "needs pyzmq")
class LiveQueryTest(unittest.TestCase):
    def test_req_rep_roundtrip(self):
        import zmq

        rep = zmq.Context.instance().socket(zmq.REP)
        port = rep.bind_to_random_port("tcp://127.0.0.1")

        def serve():
            if rep.poll(3000):
                assert rep.recv() == b"GET_DATA"
                rep.send_json(NEW)
            rep.close()

        t = threading.Thread(target=serve)
        t.start()
        self.assertEqual(resolve_camera("127.0.0.1", None, None, request_port=port), (55555, True))
        t.join()


if __name__ == "__main__":
    unittest.main()
