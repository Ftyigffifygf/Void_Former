"""Quantum gradient strategy utilities."""

from __future__ import annotations
from typing import List, Dict, Any
import torch


def compute_gradient_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Computes gradient norms separately for quantum and classical parameters."""
    classical_sq_sum = 0.0
    quantum_sq_sum = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm_sq = (param.grad.detach() ** 2).sum().item()
            if "quantum" in name.lower() or "q_layer" in name.lower() or "vqc" in name.lower() or "theta" in name.lower():
                quantum_sq_sum += norm_sq
            else:
                classical_sq_sum += norm_sq

    return {
        "classical_grad_norm": classical_sq_sum ** 0.5,
        "quantum_grad_norm": quantum_sq_sum ** 0.5,
        "total_grad_norm": (classical_sq_sum + quantum_sq_sum) ** 0.5,
    }
