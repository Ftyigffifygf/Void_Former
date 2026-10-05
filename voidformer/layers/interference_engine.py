"""Interference Engine — the core innovation.

Combines the classical state c, the void state v, and a phase-like
interference pair (i_pos, i_neg) into an emergent representation:

    combined = α·c + β·v + γ_+·i_pos + γ_-·i_neg + δ·(c ⊙ v_proj)

where α, β, γ_+, γ_-, δ are CONTEXT-ADAPTIVE — a tiny gating network
emits per-token mixing coefficients, and a learnable global scalar
sets the prior. A multiplicative term `c ⊙ v_proj` provides the
non-linear tensor-fusion component.

Output dimensionality is d_model (the classical stream's space).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterferenceEngine(nn.Module):
    def __init__(self, d_model: int, d_void: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_void = d_void

        # bring v into c-space for additive + multiplicative mixing
        self.v_to_c = nn.Linear(d_void, d_model, bias=False)

        # context-adaptive mixing coefficients (5 = α, β, γ+, γ-, δ)
        self.coef_net = nn.Sequential(
            nn.Linear(d_model + d_void, d_model),
            nn.GELU(),
            nn.Linear(d_model, 5),
        )

        # global priors (logits) — softplus -> always non-negative
        self.global_prior = nn.Parameter(torch.tensor([1.0, 0.5, 0.3, 0.3, 0.1]))

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        c: torch.Tensor,         # (B, T, d_model)
        v: torch.Tensor,         # (B, T, d_void)
        i_pos: torch.Tensor,     # (B, T, d_model)
        i_neg: torch.Tensor,     # (B, T, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_proj = self.v_to_c(v)

        # local coefficients
        local = self.coef_net(torch.cat([c, v], dim=-1))           # (B, T, 5)
        coef = F.softplus(local + self.global_prior)               # non-negative

        a, b, gp, gn, d = coef.unbind(dim=-1)                       # each (B, T)
        a = a.unsqueeze(-1); b = b.unsqueeze(-1)
        gp = gp.unsqueeze(-1); gn = gn.unsqueeze(-1); d = d.unsqueeze(-1)

        mult = c * v_proj                                           # tensor-fusion term

        combined = (
            a * c
            + b * v_proj
            + gp * i_pos
            - gn * i_neg                                            # antisymmetric phase
            + d * mult
        )
        combined = self.drop(self.norm(combined))

        # report magnitudes for stability/visualisation losses
        coefs_log = coef.detach()                                   # (B, T, 5)
        return combined, coefs_log
