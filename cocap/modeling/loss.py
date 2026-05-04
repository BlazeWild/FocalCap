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
        }
        return loss


phase2_caption_loss_cfg = builds(Phase2CaptionLoss, populate_full_signature=True)
