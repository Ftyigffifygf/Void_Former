"""Deterministic sequence prediction datasets for quantum sequence models."""

from __future__ import annotations
import torch
from torch.utils.data import TensorDataset
from typing import Tuple


def get_deterministic_sequence_dataset(
    n_samples: int = 50,
    seq_len: int = 4,
    vocab_size: int = 8,
    seed: int = 42,
) -> TensorDataset:
    """
    Generates deterministic sequence prediction tasks:
    Tokens follow a predictable arithmetic pattern: (t + start) % vocab_size
    Target is next token.
    """
    torch.manual_seed(seed)
    sequences = []
    targets = []
    for _ in range(n_samples):
        start = torch.randint(0, vocab_size, (1,)).item()
        seq = [(start + i) % vocab_size for i in range(seq_len)]
        target = [(start + i + 1) % vocab_size for i in range(seq_len)]
        sequences.append(seq)
        targets.append(target)

    x = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return TensorDataset(x, y)


def get_repeating_sequence_dataset(
    n_samples: int = 50,
    pattern: Tuple[int, ...] = (1, 2, 3),
    seq_len: int = 6,
    seed: int = 42,
) -> TensorDataset:
    """Generates repeating cyclic pattern sequence dataset."""
    torch.manual_seed(seed)
    pat_len = len(pattern)
    sequences = []
    targets = []
    for _ in range(n_samples):
        offset = torch.randint(0, pat_len, (1,)).item()
        seq = [pattern[(offset + i) % pat_len] for i in range(seq_len)]
        target = [pattern[(offset + i + 1) % pat_len] for i in range(seq_len)]
        sequences.append(seq)
        targets.append(target)

    x = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return TensorDataset(x, y)
