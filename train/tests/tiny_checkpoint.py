"""A randomly initialised, tiny NavGPT3 checkpoint with real Qwen3-VL tokenizer/processor files.

The processor files come from any Qwen3-VL checkpoint directory (no weights
needed), given by ``NAVGPT_SFT_QWEN3VL_FILES`` (default
``models/_qwen3vl_tok``), e.g. from ``hf download Qwen/Qwen3-VL-2B-Instruct
tokenizer.json tokenizer_config.json vocab.json merges.txt
preprocessor_config.json video_preprocessor_config.json config.json
chat_template.json generation_config.json``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "chat_template.json", "generation_config.json")


def qwen3vl_files() -> Path:
    return Path(os.environ.get("NAVGPT_SFT_QWEN3VL_FILES", WORKSPACE / "models" / "_qwen3vl_tok"))


def build(out_dir: Path, source: Path = None) -> Path:
    import torch

    from navgpt.vla.runtime.models.configuration_navgpt3 import NavGPT3Config
    from navgpt.vla.runtime.models.modeling_navgpt3 import NavGPT3ForConditionalGeneration

    source = Path(source or qwen3vl_files())
    base = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text = dict(base["text_config"], hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    rope = dict(text.pop("rope_scaling", None) or text.get("rope_parameters") or {}, mrope_section=[2, 3, 3])
    rope.setdefault("rope_theta", text.pop("rope_theta", 5000000))
    text["rope_parameters"] = rope
    vision = dict(base["vision_config"], depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                  out_hidden_size=64, deepstack_visual_indexes=[0])
    top = {k: base[k] for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")}
    config = NavGPT3Config(text_config=text, vision_config=vision, tie_word_embeddings=True,
                           action_hidden_size=32, **top)
    config.architectures = ["NavGPT3ForConditionalGeneration"]
    torch.manual_seed(0)
    model = NavGPT3ForConditionalGeneration(config).to(torch.bfloat16)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))

    for name in _TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, out_dir / name)
    tok_cfg = json.loads((out_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok_cfg.pop("processor_class", None)
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")
    image = json.loads((source / "preprocessor_config.json").read_text(encoding="utf-8"))
    image.update(image_processor_type="NavGPT3ImageProcessor", processor_class="NavGPT3Processor",
                 size={"shortest_edge": 4 * 32 * 32, "longest_edge": 16777216})
    (out_dir / "preprocessor_config.json").write_text(json.dumps(image, indent=2), encoding="utf-8")
    video = json.loads((source / "video_preprocessor_config.json").read_text(encoding="utf-8"))
    video.update(processor_class="NavGPT3Processor")
    (out_dir / "video_preprocessor_config.json").write_text(json.dumps(video, indent=2), encoding="utf-8")
    return out_dir
