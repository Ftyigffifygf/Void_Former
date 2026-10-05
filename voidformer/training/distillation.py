"""Distillation loss for training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> torch.Tensor:
        # Flexibly handle (student, teacher, targets) or (student, targets, teacher)
        if arg2.dtype == torch.long or arg2.ndim == 2:
            targets, teacher_logits = arg2, arg3
        else:
            teacher_logits, targets = arg2, arg3

        loss_ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
        loss_kl = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        return self.alpha * loss_ce + (1.0 - self.alpha) * loss_kl
