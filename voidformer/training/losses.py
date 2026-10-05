"""Losses for VoidFormer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class LossWeights:
    def __init__(self, ce: float = 1.0, entropy: float = 0.1):
        self.ce = ce
        self.entropy = entropy


class VoidFormerLosses:
    def __init__(self, losses_cfg: Any = None, weights: LossWeights = None, *args, **kwargs):
        self.losses_cfg = losses_cfg
        self.weights = weights or LossWeights()

    def __call__(self, out: Any, targets: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        logits = out.logits if hasattr(out, "logits") else out
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss_geo = torch.tensor(0.0, device=logits.device)
        if hasattr(out, "geodesic_distance") and out.geodesic_distance is not None:
            loss_geo = torch.mean((out.geodesic_distance - 1.0) ** 2)

        total_loss = loss_ce + loss_geo
        log_dict = {
            "total_loss": total_loss.detach(),
            "loss/geodesic": loss_geo.detach(),
        }
        return total_loss, log_dict

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
