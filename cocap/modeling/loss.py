# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch
import torch.nn.functional as F
from hydra_zen import builds
from torch import Tensor, nn


class LossBase(nn.Module):
    @abstractmethod
    def forward(self, inputs, outputs) -> Tensor:
        raise NotImplementedError


class Phase2CaptionLoss(LossBase):
    """Single CE loss over caption tokens (label smoothing=0.1 by default)."""

    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)
        self.latest_loss_components = {}

    def forward(self, target, output) -> Tensor:
        logits = output["logits"].float()            # [B, L, V]
        labels = output["labels"].long()             # [B, L]

        # Causal LM shift: logits[:, t] predicts the token at position t+1.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        zero = torch.tensor(0.0, device=loss.device)
        self.latest_loss_components = {
            "phase2/ce_loss": loss.detach(),
            "phase2/gate_mean": output.get("gate_mean", zero).detach(),
            "phase2/budget_mean": output.get("budget_mean", zero).detach(),
            "phase2/budget_std": output.get("budget_std", zero).detach(),
            "phase2/action_norm": output.get("action_norm", zero).detach(),
            "phase2/patch_diversity": output.get("patch_diversity", zero).detach(),
            # Per-block visual-token magnitudes after Modality Projector. If
            # cls_norm ≫ patch_norm, GPT-2 sees mostly CLS.
            "phase2/cls_norm_post": output.get("cls_norm_post", zero).detach(),
            "phase2/act_norm_post": output.get("act_norm_post", zero).detach(),
            "phase2/patch_norm_post": output.get("patch_norm_post", zero).detach(),
            # Caption→Visual attention mass per block. attn_*_mass should sum to
            # roughly the caption→visual attention budget; relative ratios show
            # WHO is being read. Logged only on probe steps; otherwise 0.
            "phase2/attn_cls_mass": output.get("attn_cls_mass", zero).detach(),
            "phase2/attn_act_mass": output.get("attn_act_mass", zero).detach(),
            "phase2/attn_patch_mass": output.get("attn_patch_mass", zero).detach(),
        }
        return loss


phase2_caption_loss_cfg = builds(Phase2CaptionLoss, populate_full_signature=True)
