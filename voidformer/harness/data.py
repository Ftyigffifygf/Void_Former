"""Data Context Window Optimizer for Harness Package."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class ContextWindowOptimizer:
    """Context window optimizer supporting high sequence lengths up to 8192 tokens."""

    def __init__(self, max_context_len: int = 8192, chunk_size: int = 512):
        self.max_context_len = max_context_len
        self.chunk_size = chunk_size

    def optimize_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.size(-1) > self.max_context_len:
            return token_ids[..., :self.max_context_len]
        return token_ids


class SyntheticDataset(Dataset):
    def __init__(self, vocab_size: int = 128, seq_len: int = 512, num_samples: int = 100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.data[idx]
        return {"input_ids": x, "targets": x, "labels": x}


def create_dataloader(
    vocab_size: int = 128,
    seq_len: int = 512,
    batch_size: int = 8,
    num_samples: int = 100,
) -> DataLoader:
    dataset = SyntheticDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
