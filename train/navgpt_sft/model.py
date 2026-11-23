"""NavGPT3 with LoRA on the language model, a fully trained action head, and a frozen vision tower.

The loss follows the paper (Eq. 5): mean squared error over the 24 normalised
waypoint values for navigation samples, plus next-token cross-entropy over
assistant tokens for vision-language samples. Navigation samples carry no
next-token loss and vision-language samples no waypoint loss.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from navgpt.vla.checkpoint import validate_checkpoint
from navgpt.vla.runtime.inference import default_attn_implementation
from navgpt.vla.runtime.models.modeling_navgpt3 import NavGPT3ForConditionalGeneration

from .config import LoraConfig

LANGUAGE_LAYER = r".*language_model\.layers\.\d+\.(?:self_attn|mlp)\.(?:{})"
ACTION_HEAD_FILE = "action_head.safetensors"
ADAPTER_DIR = "adapter"


def load_base(checkpoint: str, attn_implementation: str = None, dtype=torch.bfloat16,
              device_map=None) -> NavGPT3ForConditionalGeneration:
    """Load a NavGPT3 checkpoint strictly, as ``navgpt.vla.runtime.load_from_pretrained`` does."""
    validate_checkpoint(checkpoint)
    model, info = NavGPT3ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, device_map=device_map, output_loading_info=True,
        attn_implementation=attn_implementation or default_attn_implementation())
    errors = {k: v for k, v in info.items() if v}
    if errors:
        raise ValueError("Checkpoint does not match NavGPT3ForConditionalGeneration: {}".format(errors))
    return model


def lora_target_regex(cfg: LoraConfig) -> str:
    return LANGUAGE_LAYER.format("|".join(re.escape(m) for m in cfg.target_modules))


def add_lora(model: NavGPT3ForConditionalGeneration, cfg: LoraConfig):
    """Freeze everything, add adapters to the language-model decoder, unfreeze the action head."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    peft_cfg = PeftLoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                              target_modules=lora_target_regex(cfg), bias="none")
    peft_model = get_peft_model(model, peft_cfg)
    head = peft_model.get_base_model().action_head
    head.float()
    for p in head.parameters():
        p.requires_grad_(True)
    for name, p in peft_model.named_parameters():
        if p.requires_grad and "lora_" not in name and ".action_head." not in name:
            raise RuntimeError("unexpected trainable parameter {}".format(name))
    if any(p.requires_grad for p in peft_model.get_base_model().model.visual.parameters()):
        raise RuntimeError("vision tower must stay frozen")
    return peft_model


def parameter_groups(peft_model, lr_lora: float, lr_action_head: float, weight_decay: float) -> list:
    lora, head = [], []
    for name, p in peft_model.named_parameters():
        if p.requires_grad:
            (lora if "lora_" in name else head).append(p)
    return [{"params": lora, "lr": lr_lora, "weight_decay": weight_decay, "name": "lora"},
            {"params": head, "lr": lr_action_head, "weight_decay": weight_decay, "name": "action_head"}]


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class NavGPT3SFT(nn.Module):
    """Training forward pass; wrap this module (not the inner model) in DDP."""

    def __init__(self, peft_model, ntp_weight: float = 1.0):
        super().__init__()
        self.peft_model = peft_model
        self.ntp_weight = ntp_weight

    @property
    def navgpt3(self) -> NavGPT3ForConditionalGeneration:
        m = self.peft_model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def forward(self, input_ids, attention_mask, mm_token_type_ids, nav_target, nav_mask,
                pixel_values=None, image_grid_thw=None, labels=None) -> dict:
        m = self.navgpt3
        hidden = m.model(input_ids=input_ids, attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids,
                         pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False).last_hidden_state
        # Right padding: the last prompt token of each row, which is position -1 at batch size 1.
        last = attention_mask.sum(dim=1) - 1
        state = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        pred = m.action_head(state.to(m.action_head[0].weight.dtype)).float()

        mask = nav_mask.to(pred.dtype)
        n_nav = mask.sum()
        loss_nav = (((pred - nav_target) ** 2).mean(dim=1) * mask).sum() / n_nav.clamp(min=1)

        loss_ntp = pred.new_zeros(())
        if labels is not None:
            shifted = labels[:, 1:]
            sel = shifted != -100
            if sel.any():
                logits = m.lm_head(hidden[:, :-1][sel]).float()
                loss_ntp = F.cross_entropy(logits, shifted[sel])
        loss = loss_nav + self.ntp_weight * loss_ntp
        return {"loss": loss, "loss_nav": loss_nav.detach(), "loss_ntp": loss_ntp.detach(),
                "n_nav": n_nav.detach(), "pred": pred.detach()}


def save_trainable(peft_model, directory: Path) -> None:
    """Adapter (PEFT format) plus the action head; the base checkpoint is not copied."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(directory / ADAPTER_DIR))
    head = peft_model.get_base_model().action_head
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
              str(directory / ACTION_HEAD_FILE))


def load_trainable(peft_model, directory: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter = load_file(str(directory / ADAPTER_DIR / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(peft_model, adapter)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        raise ValueError("adapter has unexpected keys, e.g. {}".format(unexpected[0]))
    head = peft_model.get_base_model().action_head
    state = load_file(str(directory / ACTION_HEAD_FILE))
    head.load_state_dict({k: v.to(head[0].weight.dtype) for k, v in state.items()})
