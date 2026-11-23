import tempfile
import unittest
from pathlib import Path

from navgpt_g1d.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "g1d.yaml"


class ConfigTest(unittest.TestCase):
    def test_shipped_config_loads(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg.topics.cmd_vel, "rt/cmd_vel_no_limit")
        self.assertIs(cfg.navigator.tracker, cfg.tracker, "navigator tracks with the tracker section")

    def test_defaults_without_file(self):
        self.assertEqual(load_config(None).base.publish_hz, 30.0)

    def test_unknown_keys_rejected(self):
        for text in ("tracker:\n  vmax: 0.3\n", "trackr:\n  v_max: 0.3\n"):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "c.yaml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
