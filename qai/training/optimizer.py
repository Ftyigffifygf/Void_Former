"""Optimizer and parameter group construction."""

from __future__ import annotations
import torch
from typing import Dict, Any, List


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 0.01,
    quantum_lr: float = 0.01,
    classical_lr: float = 0.01,
    weight_decay: float = 0.0,
    optimizer_type: str = "adam",
) -> torch.optim.Optimizer:
    """Creates optimizer with separate parameter groups for quantum and classical parameters."""
    quantum_params = []
    classical_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "quantum" in name.lower() or "vqc" in name.lower() or "theta" in name.lower():
            quantum_params.append(param)
        else:
            classical_params.append(param)

    param_groups = []
    if classical_params:
        param_groups.append({"params": classical_params, "lr": classical_lr, "weight_decay": weight_decay})
    if quantum_params:
        param_groups.append({"params": quantum_params, "lr": quantum_lr, "weight_decay": 0.0})

    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    if optimizer_type.lower() == "sgd":
        return torch.optim.SGD(param_groups, lr=lr)
    return torch.optim.Adam(param_groups, lr=lr)
