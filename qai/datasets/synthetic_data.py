"""Synthetic datasets for validating quantum architectures (XOR, Parity, Regression)."""

from __future__ import annotations
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple


def get_xor_dataset(n_samples: int = 100, noise: float = 0.05, seed: int = 42) -> TensorDataset:
    """Generates 2D XOR dataset."""
    torch.manual_seed(seed)
    x = torch.rand(n_samples, 2) * 2.0 - 1.0  # [-1, 1]
    # XOR: y = 1 if x0 * x1 < 0 else 0
    y = ((x[:, 0] * x[:, 1]) < 0).long()
    if noise > 0.0:
        x = x + torch.randn_like(x) * noise
    return TensorDataset(x, y)


def get_parity_dataset(n_samples: int = 100, n_bits: int = 4, seed: int = 42) -> TensorDataset:
    """Generates n-bit parity dataset."""
    torch.manual_seed(seed)
    bits = torch.randint(0, 2, (n_samples, n_bits)).float()
    parity = (bits.sum(dim=-1) % 2).long()
    return TensorDataset(bits, parity)


def get_regression_dataset(n_samples: int = 100, in_features: int = 4, noise: float = 0.05, seed: int = 42) -> TensorDataset:
    """Generates synthetic nonlinear regression task: y = sin(x0) + cos(x1) + x2*x3."""
    torch.manual_seed(seed)
    x = torch.rand(n_samples, in_features) * 2.0 - 1.0
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1]) + 0.5 * (x[:, 2] * x[:, 3])
    if noise > 0.0:
        y = y + torch.randn_like(y) * noise
    return TensorDataset(x, y)
