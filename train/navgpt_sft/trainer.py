"""Training and offline evaluation loops (Accelerate).

One GPU: ``python -m navgpt_sft train --config ...``. Several GPUs: prefix with
``accelerate launch --num_processes N``; each rank holds the full model (DDP),
which suits LoRA. Checkpoints hold only the adapter, the action head and the
optimizer state; ``navgpt_sft merge`` turns one into a full NavGPT3 checkpoint.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from functools import partial
from pathlib import Path

import torch
import yaml

from .config import SftConfig, to_dict
from .data import (EvalSampler, MixtureDataset, MixtureSampler, NavSource, build_source, collate,
                   default_samples_per_epoch)
from .episodes import ARRIVAL_RADIUS_M, ARRIVAL_THETA_RAD
from .model import NavGPT3SFT, add_lora, count_parameters, load_base, load_trainable, parameter_groups, save_trainable

STATE_FILE = "trainer_state.pt"
CONFIG_FILE = "sft_config.yaml"


def load_processor(checkpoint: str):
    from navgpt.vla.runtime.models.processing_navgpt3 import NavGPT3Processor

    return NavGPT3Processor.from_pretrained(checkpoint)


def pad_token_id(processor) -> int:
    tok = processor.tokenizer
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def make_loader(sources, processor, sampler, batch_size: int, num_workers: int, train: bool):
    return torch.utils.data.DataLoader(
        MixtureDataset(sources, processor, train=train), batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, collate_fn=partial(collate, pad_token_id=pad_token_id(processor)),
        pin_memory=torch.cuda.is_available(), persistent_workers=False)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _arrival(wp: torch.Tensor) -> torch.Tensor:
    last = wp[:, -1]
    return (torch.hypot(last[:, 0], last[:, 1]) < ARRIVAL_RADIUS_M) & (last[:, 2].abs() < ARRIVAL_THETA_RAD)


@torch.no_grad()
def evaluate(model, loader, accelerator, scale) -> dict:
    """Waypoint errors in metres/degrees and arrival precision/recall on navigation samples."""
    was_training = model.training
    model.eval()
    scale_t = torch.tensor(scale, device=accelerator.device, dtype=torch.float32)
    # n, mse, wp1 m, wp8 m, wp1 deg, wp8 deg, arrival tp, fp, fn
    sums = torch.zeros(9, device=accelerator.device, dtype=torch.float64)
    for batch in loader:
        batch = to_device(batch, accelerator.device)
        out = model(**batch)
        keep = batch["nav_mask"]
        if not keep.any():
            continue
        pred = out["pred"][keep].view(-1, 8, 3) * scale_t
        target = batch["nav_target"][keep].view(-1, 8, 3) * scale_t
        mse = ((out["pred"][keep] - batch["nav_target"][keep]) ** 2).mean(dim=1)
        pos = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
        yaw = torch.rad2deg(torch.remainder(pred[..., 2] - target[..., 2] + math.pi, 2 * math.pi) - math.pi).abs()
        pa, ta = _arrival(pred), _arrival(target)
        sums += torch.stack([
            torch.tensor(float(keep.sum()), device=sums.device), mse.sum().double(),
            pos[:, 0].sum().double(), pos[:, -1].sum().double(), yaw[:, 0].sum().double(), yaw[:, -1].sum().double(),
            (pa & ta).sum().double(), (pa & ~ta).sum().double(), (~pa & ta).sum().double()])
    sums = accelerator.reduce(sums, reduction="sum")
    if was_training:
        model.train()
    n = max(float(sums[0]), 1.0)
    tp, fp, fn = (float(v) for v in sums[6:9])
    return {"samples": int(sums[0]), "mse": float(sums[1]) / n,
            "wp1_pos_m": float(sums[2]) / n, "wp8_pos_m": float(sums[3]) / n,
            "wp1_yaw_deg": float(sums[4]) / n, "wp8_yaw_deg": float(sums[5]) / n,
            "arrival_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "arrival_recall": tp / (tp + fn) if tp + fn else float("nan")}


class RunLog:
    def __init__(self, directory: Path, enabled: bool):
        self.enabled = enabled
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)
            self._f = open(directory / "log.jsonl", "a", encoding="utf-8")

    def write(self, **fields) -> None:
        if not self.enabled:
            return
        self._f.write(json.dumps(fields) + "\n")
        self._f.flush()
        print(" ".join("{}={}".format(k, round(v, 5) if isinstance(v, float) else v) for k, v in fields.items()),
              flush=True)

    def close(self) -> None:
        if self.enabled:
            self._f.close()


def save_checkpoint(accelerator, model, optimizer, scheduler, cfg: SftConfig, step: int, metrics: dict = None) -> Path:
    out = Path(cfg.train.output_dir) / "checkpoint-{}".format(step)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_trainable(accelerator.unwrap_model(model).peft_model, out)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step},
                   out / STATE_FILE)
        (out / CONFIG_FILE).write_text(yaml.safe_dump(to_dict(cfg), allow_unicode=True, sort_keys=False),
                                       encoding="utf-8")
        if metrics:
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _prune(Path(cfg.train.output_dir), cfg.train.save_total_limit)
    accelerator.wait_for_everyone()
    return out


def _prune(root: Path, keep: int) -> None:
    if keep <= 0:
        return
    ckpts = sorted((p for p in root.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
                   key=lambda p: int(p.name.split("-")[-1]))
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def train(cfg: SftConfig) -> Path:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import get_cosine_schedule_with_warmup

    t = cfg.train
    accelerator = Accelerator(gradient_accumulation_steps=t.grad_accum)
    set_seed(t.seed)
    out_dir = Path(t.output_dir)
    log = RunLog(out_dir, accelerator.is_main_process)
    if accelerator.is_main_process:
        (out_dir / CONFIG_FILE).write_text(yaml.safe_dump(to_dict(cfg), allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")

    processor = load_processor(cfg.model.checkpoint)
    if not cfg.data.train:
        raise ValueError("data.train is empty")
    sources = [build_source(s, cfg.nav) for s in cfg.data.train]
    for src, s in zip(sources, cfg.data.train):
        if len(src) == 0:
            raise ValueError("train source {} has no samples".format(s.manifest))
        stats = {"source": src.name, "kind": src.kind, "samples": len(src), "weight": s.weight}
        if src.kind == "nav":
            stats.update(episodes=len(src.index.episodes), **src.index.class_counts())
        log.write(event="data", **stats)
    val_sources = [NavSource(m, cfg.nav) for m in cfg.data.val]

    base = load_base(cfg.model.checkpoint, cfg.model.attn_implementation)
    base.config.use_cache = False
    if cfg.model.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = add_lora(base, cfg.lora)
    resume = Path(t.resume) if t.resume else None
    if resume:
        load_trainable(peft_model, resume)
    log.write(event="model", checkpoint=cfg.model.checkpoint, **count_parameters(peft_model))

    model = NavGPT3SFT(peft_model, t.ntp_weight)
    model.train()
    optimizer = torch.optim.AdamW(parameter_groups(peft_model, t.lr_lora, t.lr_action_head, t.weight_decay))

    micro_per_step = t.batch_size * t.grad_accum
    sampler = MixtureSampler(sources, [s.weight for s in cfg.data.train],
                             t.samples_per_epoch or default_samples_per_epoch(sources), t.seed,
                             rank=accelerator.process_index, world=accelerator.num_processes,
                             multiple_of=micro_per_step)
    steps_per_epoch = sampler.per_rank // micro_per_step
    total_steps = steps_per_epoch * t.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(t.warmup_ratio * total_steps), total_steps)
    loader = make_loader(sources, processor, sampler, t.batch_size, t.num_workers, train=True)
    val_loader = None
    if val_sources:
        val_loader = make_loader(val_sources, processor,
                                 EvalSampler(val_sources, t.seed, accelerator.process_index, accelerator.num_processes),
                                 t.batch_size, t.num_workers, train=False)

    model, optimizer = accelerator.prepare(model, optimizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    step = 0
    if resume:
        state = torch.load(resume / STATE_FILE, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
    log.write(event="plan", steps_per_epoch=steps_per_epoch, total_steps=total_steps, start_step=step,
              world=accelerator.num_processes, global_batch=micro_per_step * accelerator.num_processes)

    acc = {"loss": 0.0, "nav": 0.0, "n_nav": 0.0, "ntp": 0.0, "n_ntp": 0.0, "micro": 0}
    grad_norm = float("nan")
    metrics = None
    t_last = time.time()
    for epoch in range(step // steps_per_epoch, t.epochs):
        sampler.set_epoch(epoch, skip=max(0, step - epoch * steps_per_epoch) * micro_per_step)
        for batch in loader:
            batch = to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                out = model(**batch)
                accelerator.backward(out["loss"])
                if accelerator.sync_gradients:
                    grad_norm = float(accelerator.clip_grad_norm_(trainable, t.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            n_nav = float(out["n_nav"])
            acc["loss"] += float(out["loss"].detach())
            acc["nav"] += float(out["loss_nav"]) * n_nav
            acc["n_nav"] += n_nav
            if "labels" in batch:
                acc["ntp"] += float(out["loss_ntp"])
                acc["n_ntp"] += 1
            acc["micro"] += 1
            if not accelerator.sync_gradients:
                continue
            scheduler.step()
            step += 1
            if step % t.log_steps == 0 or step == total_steps:
                now = time.time()
                lrs = {"lr_" + g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)}
                log.write(event="train", step=step, epoch=epoch, loss=acc["loss"] / max(acc["micro"], 1),
                          loss_nav=acc["nav"] / acc["n_nav"] if acc["n_nav"] else float("nan"),
                          loss_ntp=acc["ntp"] / acc["n_ntp"] if acc["n_ntp"] else float("nan"),
                          grad_norm=grad_norm, s_per_step=(now - t_last) / t.log_steps,
                          gpu_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0,
                          **lrs)
                acc = {k: 0.0 if k != "micro" else 0 for k in acc}
                t_last = now
            metrics = None
            if val_loader is not None and t.eval_steps and (step % t.eval_steps == 0 or step == total_steps):
                metrics = evaluate(model, val_loader, accelerator, cfg.nav.scale)
                log.write(event="eval", step=step, **metrics)
            if step % t.save_steps == 0 and step != total_steps:
                save_checkpoint(accelerator, model, optimizer, scheduler, cfg, step, metrics)
            if step >= total_steps:
                break
    final = save_checkpoint(accelerator, model, optimizer, scheduler, cfg, step, metrics)
    log.write(event="done", step=step, checkpoint=str(final))
    log.close()
    return final


def evaluate_checkpoint(cfg: SftConfig, adapter: str = None, manifests: list = None) -> dict:
    """Offline metrics for the base checkpoint, or for it plus a training checkpoint's adapter."""
    from accelerate import Accelerator

    accelerator = Accelerator()
    manifests = manifests or cfg.data.val
    if not manifests:
        raise ValueError("no evaluation manifests (data.val or --manifest)")
    processor = load_processor(cfg.model.checkpoint)
    sources = [NavSource(m, cfg.nav) for m in manifests]
    base = load_base(cfg.model.checkpoint, cfg.model.attn_implementation)
    inner = base
    if adapter:
        inner = add_lora(base, cfg.lora)
        load_trainable(inner, Path(adapter))
    model = accelerator.prepare_model(NavGPT3SFT(inner), evaluation_mode=True)
    loader = make_loader(sources, processor,
                         EvalSampler(sources, cfg.train.seed, accelerator.process_index, accelerator.num_processes),
                         cfg.train.batch_size, cfg.train.num_workers, train=False)
    return evaluate(model, loader, accelerator, cfg.nav.scale)
