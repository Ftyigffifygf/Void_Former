"""Training utilities and loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class VoidFormerLosses:
    def __init__(self):
        pass

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: VoidFormerLosses = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or VoidFormerLosses()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x)
        logits = out.logits if hasattr(out, "logits") else out
        loss = self.loss_fn.compute_loss(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
