"""Embeddings and Positional Encoding modules."""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_seq_len: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to input tensor of shape (batch, seq_len, d_model)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TokenPositionalEmbedding(nn.Module):
    """Combines token lookup embedding with positional encoding and dropout."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.token_embed(x)
        x_pos = self.pos_encode(x_emb)
        return self.dropout(x_pos)
