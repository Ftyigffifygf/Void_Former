"""Interference Attention — cross-stream attention between classical and void.

Q ← classical          (what the deterministic stream is asking)
K, V ← void            (latent possibilities to draw from)
+ symmetric reverse  (Q ← void, K, V ← classical)

The two cross-attentions form a phase-like interference pair:
    I_+ = X_attend(c → v)
    I_- = X_attend(v → c)
returned together for the InterferenceEngine to fuse.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttn(nn.Module):
    def __init__(self, d_q: int, d_kv: int, d_out: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.d_out = d_out
        self.q_proj = nn.Linear(d_q, d_out, bias=False)
        self.k_proj = nn.Linear(d_kv, d_out, bias=False)
        self.v_proj = nn.Linear(d_kv, d_out, bias=False)
        self.out = nn.Linear(d_out, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, T, _ = q_in.shape
        q = self.q_proj(q_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(T, T, device=q_in.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out(out)


class InterferenceAttention(nn.Module):
    def __init__(self, d_model: int, d_void: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        # both outputs live in d_model space for downstream fusion.
        self.c_to_v = _CrossAttn(d_model, d_void, d_model, n_heads, dropout)
        self.v_to_c = _CrossAttn(d_void, d_model, d_model, n_heads, dropout)

    def forward(self, c: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i_pos = self.c_to_v(c, v)     # classical querying void
        i_neg = self.v_to_c(v, c)     # void querying classical
        return i_pos, i_neg
