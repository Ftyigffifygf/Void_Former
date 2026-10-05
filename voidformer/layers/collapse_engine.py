"""Adaptive Collapse Engine.

Decides — per token, per layer — how strongly to collapse the
super-position into a concrete classical-like representation.

τ = soft collapse strength ∈ (0, 1)

    output = (1 - τ) · combined + τ · classical

τ depends on:
    - context depth   (layer index / total layers)
    - confidence      (negative entropy of void attention)
    - semantic consistency  (cosine(c, combined))

Modes:
    soft      — continuous τ
    hard      — straight-through binarisation
    coexist   — τ kept low so multiple states survive
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCollapseEngine(nn.Module):
    def __init__(self, d_model: int, temperature_init: float = 1.0) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.log_temp = nn.Parameter(torch.tensor(float(temperature_init)).log())

    def forward(
        self,
        classical: torch.Tensor,          # (B, T, d_model)
        combined: torch.Tensor,           # (B, T, d_model)
        void_entropy: torch.Tensor,       # (B, T)
        depth_ratio: float,
        mode: str = "soft",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = combined.shape

        # cosine consistency
        cons = F.cosine_similarity(classical, combined, dim=-1).unsqueeze(-1)   # (B,T,1)
        # normalised entropy in [0,1]
        ent = void_entropy.unsqueeze(-1)
        ent = ent / (ent.amax(dim=1, keepdim=True) + 1e-6)

        feat = torch.cat([combined, cons, ent], dim=-1)
        logits = self.score(feat).squeeze(-1) / self.log_temp.exp()

        # depth bias: deeper layers → more collapse
        logits = logits + 4.0 * (depth_ratio - 0.5)
        tau = torch.sigmoid(logits).unsqueeze(-1)                                # (B,T,1)

        if mode == "hard":
            hard = (tau > 0.5).float()
            tau = hard + (tau - tau.detach())                                    # straight-through
        elif mode == "coexist":
            tau = tau * 0.3                                                       # cap

        out = (1.0 - tau) * combined + tau * classical
        return out, tau.squeeze(-1)
