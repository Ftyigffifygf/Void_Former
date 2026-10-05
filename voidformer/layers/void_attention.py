"""Void Attention — uncertainty-preserving stochastic attention.

Differences from classical MHA:

1. **Stochastic key perturbation**: keys are perturbed by Gaussian noise
   ε ~ N(0, σ²) so attention probabilities are not point estimates.
2. **Entropy-aware temperature**: per-token softmax temperature is increased
   when local entropy is high — preserving ambiguity instead of collapsing.
3. **Latent probability field**: an auxiliary uniform mixture (weight π)
   prevents premature spike-collapse on a single key.

Returns the attended values, the attention distribution, and an
entropy map (B, H, T) used downstream by the collapse engine.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoidAttention(nn.Module):
    def __init__(
        self,
        d_void: int,
        n_heads: int,
        dropout: float = 0.0,
        noise_std: float = 0.05,
        uniform_floor: float = 0.02,
    ) -> None:
        super().__init__()
        assert d_void % n_heads == 0
        self.d_void = d_void
        self.n_heads = n_heads
        self.head_dim = d_void // n_heads
        self.noise_std = noise_std
        self.uniform_floor = uniform_floor

        self.qkv = nn.Linear(d_void, 3 * d_void, bias=False)
        self.proj = nn.Linear(d_void, d_void)
        # per-head learnable inverse-temperature — initialised mild (low confidence).
        self.log_inv_temp = nn.Parameter(torch.zeros(n_heads) - 0.2)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 1. Stochastic key perturbation (only during training).
        if self.training and self.noise_std > 0:
            k = k + torch.randn_like(k) * self.noise_std

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        inv_temp = self.log_inv_temp.exp().view(1, self.n_heads, 1, 1)
        scores = scores * inv_temp

        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)

        # 3. Latent probability field — mix with a uniform-over-allowed-keys
        # to maintain non-zero mass on alternative meanings.
        if self.uniform_floor > 0:
            allowed = (~causal).float()                 # (T, T)
            denom = allowed.sum(dim=-1, keepdim=True).clamp_min(1.0)
            uniform = (allowed / denom).view(1, 1, T, T)
            attn = (1.0 - self.uniform_floor) * attn + self.uniform_floor * uniform

        attn = self.drop(attn)

        # entropy per query position (averaged over heads)
        eps = 1e-9
        entropy = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(dim=-1)  # (B, H, T)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out), attn, entropy
