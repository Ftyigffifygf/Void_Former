"""Uncertainty Memory — GRU-style residual store of unresolved void content.

Per layer ℓ, an update is computed:
    m_ℓ = (1 - z_ℓ) · m_{ℓ-1} + z_ℓ · ṽ_ℓ
where z_ℓ ∈ (0,1) gates how much current ambiguity to retain, and the
read-out is added back to the void stream so later layers can revisit it.

This module is shared across the depth dimension (recurrent over layers).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyMemory(nn.Module):
    def __init__(self, d_void: int) -> None:
        super().__init__()
        self.gate_z = nn.Linear(2 * d_void, d_void)
        self.gate_r = nn.Linear(2 * d_void, d_void)
        self.cand = nn.Linear(2 * d_void, d_void)
        self.read = nn.Linear(d_void, d_void)
        self.norm = nn.LayerNorm(d_void)

    def init_state(self, batch: int, seq: int, d: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, seq, d, device=device)

    def forward(
        self,
        v: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat([v, memory], dim=-1)
        z = torch.sigmoid(self.gate_z(cat))
        r = torch.sigmoid(self.gate_r(cat))
        cand = torch.tanh(self.cand(torch.cat([v, r * memory], dim=-1)))
        new_mem = (1 - z) * memory + z * cand
        readout = self.norm(v + self.read(new_mem))
        return readout, new_mem
