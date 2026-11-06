"""Data loading and splitting utilities."""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple


def create_train_val_loaders(
    dataset: Dataset,
    batch_size: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Splits dataset into deterministic train and validation DataLoaders."""
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
