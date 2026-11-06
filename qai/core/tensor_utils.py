"""Tensor and numerical utilities for PyTorch-PennyLane bridging."""

from __future__ import annotations
import torch
import numpy as np
from typing import Any, Union


def to_torch_tensor(data: Any, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Safely converts input data (scalars, numpy arrays, lists, tensors) to a PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, device=device)


def normalize_quantum_features(x: torch.Tensor, feature_min: float = 0.0, feature_max: float = np.pi) -> torch.Tensor:
    """
    Min-max normalizes tensor features into [feature_min, feature_max] for gate parameter encoding.
    Preserves numerical stability by adding eps.
    """
    eps = 1e-8
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    diff = x_max - x_min
    diff = torch.where(diff < eps, torch.ones_like(diff), diff)
    norm = (x - x_min) / diff
    return norm * (feature_max - feature_min) + feature_min


def check_finite_gradients(model: torch.nn.Module) -> bool:
    """Checks whether all gradients in the model are finite (no NaN or inf)."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                return False
    return True
