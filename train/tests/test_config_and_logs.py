import json
import tempfile
import unittest
from pathlib import Path

from navgpt_sft.config import load_config
from navgpt_sft.episodes import load_episodes
from navgpt_sft.g1d_logs import convert

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class ConfigTest(unittest.TestCase):
    def test_shipped_configs_load(self):
        for path in CONFIGS.glob("*.yaml"):
            cfg = load_config(str(path))
            self.assertTrue(cfg.data.train, path.name)

    def test_unknown_keys_and_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.yaml"
            path.write_text("train:\n  epochs: 2\n  bogus: 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(path))
            path.write_text("data:\n  train:\n    - {manifest: a.jsonl}\n", encoding="utf-8")
            cfg = load_config(str(path), ("train.epochs=7", "lora.r=8", "train.lr_lora=1e-4"))
            self.assertEqual((cfg.train.epochs, cfg.lora.r), (7, 8))
            self.assertEqual(cfg.train.lr_lora, 1e-4, "YAML reads 1e-4 as a string")
            with self.assertRaises(ValueError):
                load_config(str(path), ("train.lr_lora=fast",))
            self.assertEqual(cfg.data.train[0].kind, "nav")


class G1dLogsTest(unittest.TestCase):
    def _run(self, root: Path, name: str, status: str, seqs):
        run = root / name
        (run / "frames").mkdir(parents=True)
        events = [{"event": "start", "instruction": "Go to the door."}]
        for k, seq in enumerate(seqs):
            (run / "frames" / "{:05d}.jpg".format(seq)).write_bytes(b"jpg")
            events.append({"event": "reply", "seq": seq, "commit": True, "capture_pose": [0.25 * k, 0.0, 0.0]})
            events.append({"event": "reply", "seq": seq + 100, "commit": False, "capture_pose": [9, 9, 9]})
        events.append({"event": "result", "status": status})
        (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        return run

    def test_convert_keeps_arrived_runs_with_committed_frames(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "logs"
            runs = [self._run(root, "r1", "arrived", [1, 3, 6]), self._run(root, "r2", "stuck", [1, 2])]
            manifest = Path(d) / "data" / "g1d.jsonl"
            counts = convert(runs, str(manifest))
            self.assertEqual(counts["written"], 1)
            self.assertEqual(counts["skipped_status"], 1)
            eps = load_episodes(str(manifest))
            self.assertEqual([f.pose[0] for f in eps[0].frames], [0.0, 0.25, 0.5])
            self.assertTrue(Path(eps[0].frames[0].images[0]).exists())
            self.assertEqual(eps[0].instructions, ["Go to the door."])


if __name__ == "__main__":
    unittest.main()
