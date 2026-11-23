"""Model inputs for navigation and vision-language samples, and their mixture.

Navigation samples go through the same path as ``NavGPTVLA`` with
``frame_cache="on_demand"``: the upstream prompt template and history labels,
``build_prompt``, and the NavGPT3 processor with per-image token budgets. Their
target is the normalised 24-dimensional waypoint vector.

Vision-language samples (``kind: vl``) train the next-token objective only::

    {"images": ["img/0001.jpg"],
     "conversations": [{"role": "user", "content": "<image>\\nWhere is the trash can?"},
                       {"role": "assistant", "content": "Left of the door."}]}

Labels cover the assistant turns; prompt and padding tokens are masked.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageEnhance

from navgpt.vla.agent import _DEFAULT_CAMERA_NAMES, _DEFAULT_CAMERA_WEIGHTS, NavGPTVLA
from navgpt.vla.runtime.inference import build_prompt

from .config import NavConfig, SourceConfig
from .episodes import NavIndex, load_episodes, make_sample, normalize

IGNORE_INDEX = -100
VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def camera_names(cfg: NavConfig) -> list:
    return list(cfg.camera_names or _DEFAULT_CAMERA_NAMES.get(cfg.view_num, [f"View {i}" for i in range(cfg.view_num)]))


def camera_weights(cfg: NavConfig) -> list:
    return list(cfg.camera_weights or _DEFAULT_CAMERA_WEIGHTS.get(cfg.view_num, [1.0] * cfg.view_num))


def nav_prompt(instruction: str, history_steps: list, names: list) -> str:
    describe = NavGPTVLA._build_image_describe(SimpleNamespace(camera_names=names), history_steps)
    return build_prompt(NavGPTVLA.PROMPT_TEMPLATE.format(describe, instruction))


def _jitter(images: list, rng: np.random.Generator, amount: float) -> list:
    if amount <= 0:
        return images
    factors = rng.uniform(1 - amount, 1 + amount, size=3)
    out = []
    for img in images:
        img = ImageEnhance.Brightness(img).enhance(factors[0])
        img = ImageEnhance.Contrast(img).enhance(factors[1])
        out.append(ImageEnhance.Color(img).enhance(factors[2]))
    return out


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class NavSource:
    kind = "nav"

    def __init__(self, manifest: str, cfg: NavConfig, name: str = None):
        self.name = name or Path(manifest).stem
        self.cfg = cfg
        episodes = load_episodes(manifest, cfg.view_num)
        self.index = NavIndex.build(episodes, cfg.keyframe_m, cfg.keyframe_rad)
        self.names = camera_names(cfg)
        self.weights = camera_weights(cfg)

    def __len__(self) -> int:
        return len(self.index.items)

    def item_weights(self) -> np.ndarray:
        w = np.asarray([self.cfg.resample.get(c, 1.0) for c in self.index.classes], dtype=np.float64)
        return w / w.sum()

    def budget(self, rng: np.random.Generator, train: bool) -> dict:
        c = self.cfg
        if train and rng.random() < c.p_random_budget:
            return {"max_nav_vis_tokens": int(rng.integers(c.random_total_tokens[0], c.random_total_tokens[1] + 1)),
                    "temporal_decay": float(rng.uniform(*c.random_temporal_decay)),
                    "min_tokens_per_image": c.min_tokens_per_image,
                    "max_tokens_per_image": int(rng.integers(c.random_max_tokens_per_image[0],
                                                             c.random_max_tokens_per_image[1] + 1))}
        return {"max_nav_vis_tokens": c.max_nav_vis_tokens, "temporal_decay": c.temporal_decay,
                "min_tokens_per_image": c.min_tokens_per_image, "max_tokens_per_image": c.max_tokens_per_image}

    def encode(self, processor, item: int, rng: np.random.Generator, train: bool) -> dict:
        c = self.cfg
        sample = make_sample(self.index, item, rng, c.keyframe_m, c.keyframe_rad, c.max_sample_size,
                             p_latest=c.p_latest if train else 0.0,
                             p_off_keyframe=c.p_off_keyframe if train else 0.0)
        images = [_load_rgb(p) for frame in sample.history for p in frame.images]
        if train:
            images = _jitter(images, rng, c.color_jitter)
        inputs = processor(
            text=nav_prompt(sample.instruction, sample.history_steps, self.names),
            images=images, view_num=c.view_num, steps=len(sample.history),
            camera_weights=self.weights, codec_enabled=c.codec_enabled, return_tensors="pt",
            **self.budget(rng, train),
        )
        out = {k: inputs[k][0] for k in ("input_ids", "attention_mask", "mm_token_type_ids")}
        out["pixel_values"], out["image_grid_thw"] = inputs["pixel_values"], inputs["image_grid_thw"]
        out["nav_target"] = torch.tensor(normalize(sample.waypoints, c.scale), dtype=torch.float32)
        out["nav_mask"] = torch.tensor(True)
        return out


class VlSource:
    kind = "vl"

    def __init__(self, manifest: str, cfg: NavConfig, name: str = None):
        self.name = name or Path(manifest).stem
        self.cfg = cfg
        root = Path(manifest).parent
        self.records = []
        with open(manifest, encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                images = [str(p if Path(p).is_absolute() else root / p) for p in raw.get("images", [])]
                turns = raw["conversations"]
                placeholders = sum(t["content"].count("<image>") for t in turns)
                if placeholders != len(images):
                    raise ValueError("{}:{}: {} <image> placeholder(s) for {} image(s)".format(
                        manifest, line_no, placeholders, len(images)))
                if not any(t["role"] == "assistant" for t in turns):
                    raise ValueError("{}:{}: no assistant turn to supervise".format(manifest, line_no))
                self.records.append((images, turns))

    def __len__(self) -> int:
        return len(self.records)

    def item_weights(self) -> np.ndarray:
        return np.full(len(self.records), 1.0 / len(self.records))

    def encode(self, processor, item: int, rng: np.random.Generator, train: bool) -> dict:
        image_paths, turns = self.records[item]
        text = "".join("<|im_start|>{}\n{}<|im_end|>\n".format(t["role"], t["content"].replace("<image>", VISION_PLACEHOLDER))
                       for t in turns)
        kwargs = {}
        images = None
        if image_paths:
            images = [_load_rgb(p) for p in image_paths]
            n, cap = len(images), self.cfg.vl_tokens_per_image
            kwargs = dict(view_num=n, steps=1, max_nav_vis_tokens=n * cap, min_tokens_per_image=4,
                          max_tokens_per_image=cap, codec_enabled=False)
        inputs = processor(text=text, images=images, return_tensors="pt", **kwargs)
        out = {k: inputs[k][0] for k in ("input_ids", "attention_mask", "mm_token_type_ids")}
        if images is not None:
            out["pixel_values"], out["image_grid_thw"] = inputs["pixel_values"], inputs["image_grid_thw"]
        out["labels"] = assistant_labels(out["input_ids"], processor.tokenizer)
        out["nav_target"] = torch.zeros(24, dtype=torch.float32)
        out["nav_mask"] = torch.tensor(False)
        return out


def assistant_labels(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Copy of ``input_ids`` with everything but assistant replies (and their <|im_end|>) masked."""
    ids = input_ids.tolist()
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    header = tokenizer.encode("assistant\n", add_special_tokens=False)
    labels = [IGNORE_INDEX] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] == im_start and ids[i + 1:i + 1 + len(header)] == header:
            start = i + 1 + len(header)
            end = start
            while end < len(ids) and ids[end] != im_end:
                end += 1
            stop = min(end + 1, len(ids))
            labels[start:stop] = ids[start:stop]
            i = stop
        else:
            i += 1
    return torch.tensor(labels, dtype=torch.long)


def build_source(src: SourceConfig, cfg: NavConfig):
    cls = NavSource if src.kind == "nav" else VlSource
    return cls(src.manifest, cfg, src.name)


class MixtureDataset(torch.utils.data.Dataset):
    """Indexed by ``(source, item, seed)`` tuples from ``MixtureSampler``."""

    def __init__(self, sources: list, processor, train: bool = True):
        self.sources, self.processor, self.train = sources, processor, train

    def __len__(self) -> int:
        return sum(len(s) for s in self.sources)

    def __getitem__(self, key) -> dict:
        source, item, seed = key
        rng = np.random.default_rng(seed)
        return self.sources[source].encode(self.processor, item, rng, self.train)


class MixtureSampler(torch.utils.data.Sampler):
    """Per-epoch draws over sources by weight and over items by motion class.

    Every rank derives the same draw from ``(seed, epoch)`` and keeps its
    stride, so no DistributedSampler is needed.
    """

    def __init__(self, sources: list, weights: list, num_samples: int, seed: int,
                 rank: int = 0, world: int = 1, multiple_of: int = 1):
        self.sources = sources
        w = np.asarray(weights, dtype=np.float64)
        self.source_p = w / w.sum()
        self.item_p = [s.item_weights() for s in sources]
        self.seed, self.rank, self.world = seed, rank, world
        per_rank = num_samples // world
        self.per_rank = max(multiple_of, per_rank // multiple_of * multiple_of)
        self.epoch = 0
        self.skip = 0                         # draws of this epoch already consumed (resume)

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = epoch, skip

    def __len__(self) -> int:
        return self.per_rank - self.skip

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        total = self.per_rank * self.world
        src = rng.choice(len(self.sources), size=total, p=self.source_p)
        items = np.empty(total, dtype=np.int64)
        for s in range(len(self.sources)):
            sel = np.nonzero(src == s)[0]
            if len(sel):
                items[sel] = rng.choice(len(self.sources[s]), size=len(sel), p=self.item_p[s])
        seeds = rng.integers(0, 2 ** 63 - 1, size=total)
        for i in range(self.rank + self.skip * self.world, total, self.world):
            yield int(src[i]), int(items[i]), int(seeds[i])


class EvalSampler(torch.utils.data.Sampler):
    """Every item of every source once, sharded across ranks, with fixed seeds."""

    def __init__(self, sources: list, seed: int = 0, rank: int = 0, world: int = 1):
        self.keys = [(s, i, seed * 1_000_003 + s * 100_003 + i)
                     for s, src in enumerate(sources) for i in range(len(src))][rank::world]

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self):
        return iter(self.keys)


def collate(batch: list, pad_token_id: int) -> dict:
    """Right-pad token tensors; concatenate image patches and grids along dim 0."""
    length = max(len(b["input_ids"]) for b in batch)

    def pad(key, value):
        out = torch.full((len(batch), length), value, dtype=batch[0][key].dtype)
        for i, b in enumerate(batch):
            out[i, :len(b[key])] = b[key]
        return out

    out = {
        "input_ids": pad("input_ids", pad_token_id),
        "attention_mask": pad("attention_mask", 0),
        "mm_token_type_ids": pad("mm_token_type_ids", 0),
        "nav_target": torch.stack([b["nav_target"] for b in batch]),
        "nav_mask": torch.stack([b["nav_mask"] for b in batch]),
    }
    if any("labels" in b for b in batch):
        for b in batch:
            b.setdefault("labels", torch.full_like(b["input_ids"], IGNORE_INDEX))
        out["labels"] = pad("labels", IGNORE_INDEX)
    with_images = [b for b in batch if "pixel_values" in b]
    if with_images:
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in with_images])
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in with_images])
    return out


def default_samples_per_epoch(sources: list) -> int:
    return sum(len(s) for s in sources)


def steps_per_epoch(samples_per_rank: int, batch_size: int, grad_accum: int) -> int:
    return max(1, math.floor(samples_per_rank / (batch_size * grad_accum)))
