"""End to end on CPU with a tiny random NavGPT3: train, evaluate, resume, merge, load strictly.

Skipped unless torch, transformers, peft and accelerate are installed and the
Qwen3-VL processor files of tests/tiny_checkpoint.py are present.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tiny_checkpoint

_MISSING = [m for m in ("torch", "transformers", "peft", "accelerate", "torchvision") if importlib.util.find_spec(m) is None]
_REASON = "needs {}".format(", ".join(_MISSING)) if _MISSING else \
    "needs Qwen3-VL processor files in {}".format(tiny_checkpoint.qwen3vl_files())


def _write_episode(root: Path, name: str, poses, rng, size=128):
    from PIL import Image

    (root / name).mkdir(parents=True, exist_ok=True)
    frames = []
    for i, pose in enumerate(poses):
        path = root / name / "{:03d}.jpg".format(i)
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path)
        frames.append({"image": "{}/{:03d}.jpg".format(name, i), "pose": [float(v) for v in pose]})
    return {"episode_id": name, "instructions": ["Walk forward and stop.", "Go ahead, then stop."], "frames": frames}


@unittest.skipIf(_MISSING or not (tiny_checkpoint.qwen3vl_files() / "tokenizer.json").exists(), _REASON)
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ckpt = tiny_checkpoint.build(root / "tiny")
        data = root / "data"
        rng = np.random.default_rng(0)
        straight = [(0.05 * i, 0.0, 0.0) for i in range(40)]
        left = [(0.05 * i, 0.0, 0.03 * i) for i in range(30)]
        lines = [json.dumps(_write_episode(data, n, p, rng)) for n, p in (("s", straight), ("l", left))]
        (data / "nav.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        from PIL import Image

        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(data / "vl.jpg")
        vl = {"images": ["vl.jpg"], "conversations": [
            {"role": "user", "content": "<image>\nWhere is the trash can?"},
            {"role": "assistant", "content": "Left of the door."}]}
        (data / "vl.jsonl").write_text(json.dumps(vl) + "\n", encoding="utf-8")
        cls.data = data

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def config(self, *overrides):
        from navgpt_sft.config import load_config

        base = (
            "model.checkpoint={}".format(self.ckpt.as_posix()),
            "model.attn_implementation=sdpa",
            "data.train=[{{manifest: {0}/nav.jsonl, weight: 0.8}}, {{manifest: {0}/vl.jsonl, kind: vl, weight: 0.2}}]"
            .format(self.data.as_posix()),
            "data.val=[{}/nav.jsonl]".format(self.data.as_posix()),
            "train.output_dir={}".format((Path(self.tmp.name) / "run").as_posix()),
            "train.samples_per_epoch=8", "train.grad_accum=2", "train.num_workers=0",
            "train.log_steps=1", "train.eval_steps=2", "train.save_steps=2", "train.save_total_limit=5",
            "train.lr_lora=5e-3", "train.lr_action_head=5e-3", "train.warmup_ratio=0.0",
        )
        return load_config(None, base + overrides)

    def test_vl_labels_cover_only_the_answer(self):
        from navgpt_sft.data import IGNORE_INDEX, VlSource
        from navgpt_sft.trainer import load_processor

        cfg = self.config()
        processor = load_processor(cfg.model.checkpoint)
        item = VlSource(str(self.data / "vl.jsonl"), cfg.nav).encode(processor, 0, np.random.default_rng(0), True)
        answer = item["input_ids"][item["labels"] != IGNORE_INDEX]
        self.assertEqual(processor.tokenizer.decode(answer), "Left of the door.<|im_end|>")
        self.assertIn("pixel_values", item)

    def test_train_eval_resume_merge(self):
        import torch

        from navgpt.vla.runtime import load_from_pretrained
        from navgpt.vla.runtime.inference import MODEL_INPUTS
        from navgpt_sft.data import NavSource, collate
        from navgpt_sft.export import merge
        from navgpt_sft.model import NavGPT3SFT, add_lora, load_base, load_trainable
        from navgpt_sft.trainer import evaluate_checkpoint, load_processor, train

        first = train(self.config("train.epochs=1"))
        self.assertEqual(first.name, "checkpoint-4")
        run = first.parent
        events = [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = {e["event"] for e in events}
        self.assertTrue({"data", "model", "plan", "train", "eval", "done"} <= kinds)
        losses = [e["loss"] for e in events if e["event"] == "train"]
        self.assertTrue(all(np.isfinite(losses)))
        self.assertTrue((run / "checkpoint-2" / "adapter" / "adapter_model.safetensors").exists())

        resumed = train(self.config("train.epochs=2", "train.resume={}".format((run / "checkpoint-2").as_posix())))
        self.assertEqual(resumed.name, "checkpoint-8")
        plan = [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()
                if '"plan"' in line][-1]
        self.assertEqual(plan["start_step"], 2)

        cfg = self.config()
        base_metrics = evaluate_checkpoint(cfg)
        tuned_metrics = evaluate_checkpoint(cfg, adapter=str(resumed))
        self.assertEqual(base_metrics["samples"], tuned_metrics["samples"])
        self.assertNotAlmostEqual(base_metrics["mse"], tuned_metrics["mse"], places=6)

        merged_dir = Path(self.tmp.name) / "merged"
        merge(str(self.ckpt), str(resumed), str(merged_dir))
        model, processor = load_from_pretrained(str(merged_dir), device="cpu", attn_implementation="sdpa")
        self.assertTrue((merged_dir / "sft_provenance.json").exists())

        item = NavSource(str(self.data / "nav.jsonl"), cfg.nav).encode(processor, 3, np.random.default_rng(5), False)
        inputs = {k: item[k].unsqueeze(0) if item[k].dim() == 1 else item[k] for k in MODEL_INPUTS}
        adapted = add_lora(load_base(str(self.ckpt), "sdpa"), cfg.lora)
        load_trainable(adapted, resumed)
        with torch.no_grad():
            merged_pred = model.predict_actions(**inputs).float()
            adapter_pred = NavGPT3SFT(adapted).eval()(**collate([item], processor.tokenizer.pad_token_id))["pred"]
            base_pred = load_base(str(self.ckpt), "sdpa").predict_actions(**inputs).float()
        self.assertLess(float((merged_pred - adapter_pred).abs().max()), 0.05, "merge reproduces the adapter")
        self.assertGreater(float((merged_pred - base_pred).abs().max()), 1e-3, "training changed the model")


if __name__ == "__main__":
    unittest.main()
