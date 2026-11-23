"""Merge a training checkpoint into a full NavGPT3 checkpoint.

The result has the same tensor names, configuration and processor files as the
base checkpoint, so ``navgpt.vla.runtime.load_from_pretrained`` (and therefore
``deploy/g1d/server/serve_vla.py --checkpoint``) loads it unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from navgpt.vla.checkpoint import validate_checkpoint

from .model import ACTION_HEAD_FILE, ADAPTER_DIR, load_base

_WEIGHT_FILES = ("model.safetensors.index.json",)


def merge(base_checkpoint: str, training_checkpoint: str, out_dir: str, device: str = "cpu",
          max_shard_size: str = "5GB") -> dict:
    from peft import PeftModel
    from safetensors.torch import load_file

    base_dir, ckpt, out = Path(base_checkpoint), Path(training_checkpoint), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("{} is not empty".format(out))
    model = load_base(str(base_dir), attn_implementation="sdpa", device_map=device)
    model = PeftModel.from_pretrained(model, str(ckpt / ADAPTER_DIR)).merge_and_unload()
    head = load_file(str(ckpt / ACTION_HEAD_FILE))
    dtype = model.action_head[0].weight.dtype
    model.action_head.load_state_dict({k: v.to(dtype) for k, v in head.items()})

    model.save_pretrained(str(out), max_shard_size=max_shard_size)
    # Configuration, tokenizer and processor files come from the base checkpoint
    # verbatim; the architecture is unchanged.
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in _WEIGHT_FILES:
            shutil.copy2(f, out / f.name)
    for src, dst in (("metrics.json", "sft_metrics.json"), ("sft_config.yaml", "sft_config.yaml")):
        if (ckpt / src).exists():
            shutil.copy2(ckpt / src, out / dst)
    provenance = {"base_checkpoint": str(base_dir), "training_checkpoint": str(ckpt)}
    (out / "sft_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return validate_checkpoint(str(out))
