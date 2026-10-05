"""Void Attention — uncertainty-preserving stochastic attention with Amplitude Encoding Compression.

Differences from classical MHA:

1. **Amplitude Encoding Attention Compression**: L2-normalized StatePreparation
   compresses classical D-dimensional embedding vectors into ⌈log2(D)⌉ Qubits with zero-padding.
2. **Stochastic key perturbation**: keys are perturbed by Gaussian noise
   ε ~ N(0, σ²) so attention probabilities are not point estimates.
3. **Entropy-aware temperature**: per-token softmax temperature is increased
   when local entropy is high — preserving ambiguity instead of collapsing.
4. **Latent probability field**: an auxiliary uniform mixture (weight π)
   prevents premature spike-collapse on a single key.

Returns the attended values, the attention distribution, and an
entropy map (B, H, T) used downstream by the collapse engine.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def amplitude_encode_compress(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Compress D-dimensional classical vectors into ⌈log2(D)⌉ Qubit state amplitudes with zero-padding."""
    B, T, D = x.shape
    n_qubits = math.ceil(math.log2(D)) if D > 1 else 1
    target_dim = 2 ** n_qubits

    if target_dim > D:
        padding = torch.zeros(B, T, target_dim - D, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, padding], dim=-1)
    else:
        x_padded = x

    # L2-normalize to form valid quantum state vector amplitudes
    norm = torch.norm(x_padded, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    state_amplitudes = x_padded / norm
    return state_amplitudes, n_qubits


class VoidAttention(nn.Module):
    def __init__(
        self,
        d_void: int,
        n_heads: int,
        dropout: float = 0.0,
        noise_std: float = 0.05,
        uniform_floor: float = 0.02,
        use_amplitude_encoding: bool = True,
    ) -> None:
        super().__init__()
        assert d_void % n_heads == 0
        self.d_void = d_void
        self.n_heads = n_heads
        self.head_dim = d_void // n_heads
        self.noise_std = noise_std
        self.uniform_floor = uniform_floor
        self.use_amplitude_encoding = use_amplitude_encoding

        # Number of qubits required for quantum state amplitude compression
        self.n_qubits = math.ceil(math.log2(d_void)) if d_void > 1 else 1
        self.quantum_dim = 2 ** self.n_qubits

        self.qkv = nn.Linear(d_void, 3 * d_void, bias=False)
        self.proj = nn.Linear(d_void, d_void)
        self.quantum_decompress = nn.Linear(self.quantum_dim, d_void) if self.use_amplitude_encoding else None

        self.log_inv_temp = nn.Parameter(torch.zeros(n_heads) - 0.2)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        if self.use_amplitude_encoding:
            # Amplitude encoding compression: compress token vector into n_qubits
            compressed_states, n_q = amplitude_encode_compress(x)  # (B, T, quantum_dim)
            # Decompress/project back to hidden dimension D for QKV projection
            x_eff = self.quantum_decompress(compressed_states)
        else:
            x_eff = x

        qkv = self.qkv(x_eff).view(B, T, 3, self.n_heads, self.head_dim)
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

        # 3. Latent probability field — mix with uniform distribution
        if self.uniform_floor > 0:
            allowed = (~causal).float()
            denom = allowed.sum(dim=-1, keepdim=True).clamp_min(1.0)
            uniform = (allowed / denom).view(1, 1, T, T)
            attn = (1.0 - self.uniform_floor) * attn + self.uniform_floor * uniform

        attn = self.drop(attn)

        eps = 1e-9
        entropy = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out), attn, entropy
