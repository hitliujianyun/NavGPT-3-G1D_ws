"""One YAML file configures a fine-tuning run; unknown keys are rejected.

Relative paths are resolved against the current working directory, so run the
commands from the workspace root.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    checkpoint: str = "models/NavGPT3-4B"     # NavGPT3 checkpoint to fine-tune
    attn_implementation: Optional[str] = None  # sdpa | flash_attention_2 (default: auto)
    gradient_checkpointing: bool = True


@dataclass
class LoraConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    # Linear layers of the language-model decoder that receive adapters. The
    # vision tower, embeddings and LM head stay frozen; the action head is
    # trained in full.
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


@dataclass
class SourceConfig:
    manifest: str = ""
    kind: str = "nav"                         # nav: waypoint regression | vl: next-token only
    weight: float = 1.0                       # share of the epoch drawn from this source
    name: Optional[str] = None


@dataclass
class DataConfig:
    train: list = field(default_factory=list)  # SourceConfig entries
    val: list = field(default_factory=list)    # nav manifests for evaluation


@dataclass
class NavConfig:
    view_num: int = 1                         # G1-D head camera; 4 for front/right/back/left
    camera_names: Optional[list] = None       # default: navgpt.vla.agent names for view_num
    camera_weights: Optional[list] = None     # default: navgpt.vla.agent weights for view_num
    keyframe_m: float = 0.25                  # navgpt_g1d navigator.keyframe_m
    keyframe_rad: float = math.radians(15)    # navgpt_g1d navigator.keyframe_rad
    scale: list = field(default_factory=lambda: [1.0, 0.433, math.pi / 6 * 4])  # habitat_nav
    max_sample_size: int = 16
    p_latest: float = 0.0                     # share of "latest" history sampling (serve_vla: random)
    p_off_keyframe: float = 0.5               # current frame between keyframes (needs dense frames)
    # Visual token budget at deployment (serve_vla / NavGPTVLA defaults) ...
    max_nav_vis_tokens: int = 3072
    temporal_decay: float = 2.0
    min_tokens_per_image: int = 4
    max_tokens_per_image: int = 196
    codec_enabled: bool = True
    # ... and how often, and over which ranges, it is randomised instead.
    p_random_budget: float = 0.3
    random_total_tokens: list = field(default_factory=lambda: [1024, 4096])
    random_temporal_decay: list = field(default_factory=lambda: [1.0, 3.0])
    random_max_tokens_per_image: list = field(default_factory=lambda: [128, 256])
    color_jitter: float = 0.1                 # brightness/contrast/saturation +-; 0 disables
    resample: dict = field(default_factory=lambda: {"stop": 2.0, "turn": 2.0, "straight": 1.0})
    vl_tokens_per_image: int = 256            # cap for images in vl samples


@dataclass
class TrainConfig:
    output_dir: str = "outputs/sft"
    seed: int = 0
    epochs: int = 3
    samples_per_epoch: Optional[int] = None   # default: size of all train sources
    batch_size: int = 1                       # per device
    grad_accum: int = 16
    lr_lora: float = 1e-4
    lr_action_head: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    ntp_weight: float = 1.0                   # next-token loss weight for vl samples
    num_workers: int = 4
    log_steps: int = 10
    eval_steps: int = 200                     # optimizer steps; 0 disables
    save_steps: int = 500
    save_total_limit: int = 3
    resume: Optional[str] = None              # checkpoint-<step> directory to continue from


@dataclass
class SftConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    nav: NavConfig = field(default_factory=NavConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS = {"model": ModelConfig, "lora": LoraConfig, "data": DataConfig, "nav": NavConfig, "train": TrainConfig}


def _build(cls, data: dict, where: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError("unknown key(s) {} in {}; valid: {}".format(sorted(unknown), where, sorted(names)))
    hints = typing.get_type_hints(cls)
    return cls(**{k: _coerce(v, hints[k], "{}.{}".format(where, k)) for k, v in data.items()})


def _coerce(value, hint, where: str):
    """YAML reads ``1e-4`` (no decimal point) as a string; convert numbers by field type."""
    args = typing.get_args(hint)
    target = next((a for a in args if a is not type(None)), hint) if args else hint
    if value is None or target not in (int, float) or isinstance(value, bool):
        return value
    try:
        return target(value) if target is float else int(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a number, got {!r}".format(where, value)) from None


def _parse_override(item: str):
    if "=" not in item:
        raise ValueError("override must look like section.key=value, got {!r}".format(item))
    key, value = item.split("=", 1)
    if key.count(".") != 1:
        raise ValueError("override key must be section.key, got {!r}".format(key))
    section, name = key.split(".")
    return section, name, yaml.safe_load(value)


def load_config(path: Optional[str] = None, overrides: tuple = ()) -> SftConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    raw = raw or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError("unknown section(s) {}; valid: {}".format(sorted(unknown), sorted(_SECTIONS)))
    for item in overrides:
        section, name, value = _parse_override(item)
        raw.setdefault(section, {})
        raw[section] = dict(raw[section] or {}, **{name: value})
    built = {name: _build(cls, raw.get(name) or {}, name) for name, cls in _SECTIONS.items()}
    built["data"].train = [_build(SourceConfig, s, "data.train[{}]".format(i)) for i, s in enumerate(built["data"].train)]
    cfg = SftConfig(**built)
    _check(cfg)
    return cfg


def _check(cfg: SftConfig) -> None:
    for s in cfg.data.train:
        if s.kind not in ("nav", "vl"):
            raise ValueError("data.train kind must be nav or vl, got {!r}".format(s.kind))
        if not s.manifest:
            raise ValueError("data.train entries need a manifest")
        if s.weight <= 0:
            raise ValueError("data.train weight must be positive")
    if len(cfg.nav.scale) != 3:
        raise ValueError("nav.scale must be [x, y, theta]")
    if set(cfg.nav.resample) - {"stop", "turn", "straight"}:
        raise ValueError("nav.resample keys are stop, turn, straight")
    for name in ("camera_names", "camera_weights"):
        value = getattr(cfg.nav, name)
        if value is not None and len(value) != cfg.nav.view_num:
            raise ValueError("nav.{} needs view_num={} entries".format(name, cfg.nav.view_num))
    if cfg.nav.min_tokens_per_image * cfg.nav.max_sample_size * cfg.nav.view_num > min(
            cfg.nav.max_nav_vis_tokens, cfg.nav.random_total_tokens[0]):
        raise ValueError("nav.min_tokens_per_image x max_sample_size x view_num exceeds the token budget")


def to_dict(cfg: SftConfig) -> dict:
    return dataclasses.asdict(cfg)
