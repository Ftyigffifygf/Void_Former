"""Training metrics and evaluation measurements."""

from __future__ import annotations
import torch
import math
from typing import Dict, Any


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, task: str = "classification") -> Dict[str, float]:
    """Computes relevant metrics based on the task."""
    metrics = {}
    if task == "classification":
        if preds.dim() > 1 and preds.size(-1) > 1:
            pred_classes = torch.argmax(preds, dim=-1)
        else:
            pred_classes = (torch.sigmoid(preds) > 0.5).long()

        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        accuracy = correct / max(total, 1)
        metrics["accuracy"] = accuracy

    elif task == "sequence_modeling":
        # preds: [batch, seq_len, vocab_size], targets: [batch, seq_len]
        pred_tokens = torch.argmax(preds, dim=-1)
        correct = (pred_tokens == targets).sum().item()
        total = targets.numel()
        accuracy = correct / max(total, 1)
        metrics["token_accuracy"] = accuracy

    elif task == "regression":
        # MSE & MAE
        mse = torch.mean((preds.squeeze() - targets.squeeze()) ** 2).item()
        mae = torch.mean(torch.abs(preds.squeeze() - targets.squeeze())).item()
        metrics["mse"] = mse
        metrics["mae"] = mae

    return metrics
