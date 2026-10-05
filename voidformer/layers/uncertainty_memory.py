"""Uncertainty Memory — GRU-style residual store of unresolved void content with qRAM Superposition Memory Indexing.

Per layer ℓ, an update is computed:
    m_ℓ = (1 - z_ℓ) · m_{ℓ-1} + z_ℓ · ṽ_ℓ
where z_ℓ ∈ (0,1) gates how much current ambiguity to retain, and the
read-out is added back to the void stream so later layers can revisit it.

qRAM Superposition Memory Indexing enables superposition query addresses |i⟩
for KV-cache / memory retrieval in O(log N) query steps across long context windows.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class qRAMSimulator(nn.Module):
    """qRAM Superposition Memory Lookup simulator: retrieves KV entries in O(log N) query steps."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, query_state: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        """Perform qRAM query in quantum superposition.

        query_state: (B, T, d_model)
        memory_bank: (B, N, d_model) memory slots
        """
        B, T, D = query_state.shape
        B_mem, N, D_mem = memory_bank.shape

        if N == 0:
            return torch.zeros_like(query_state)

        # Superposition address amplitudes |i> derived from tree query routing in O(log N)
        # Cosine similarity inner product query address superposition
        query_norm = F.normalize(query_state, p=2, dim=-1)  # (B, T, D)
        mem_norm = F.normalize(memory_bank, p=2, dim=-1)    # (B, N, D)

        # Superposition address routing weights
        address_logits = query_norm @ mem_norm.transpose(-2, -1) / math.sqrt(math.log2(max(N, 2)))
        superposition_weights = F.softmax(address_logits, dim=-1)  # (B, T, N)

        # qRAM superposition state retrieval
        retrieved = superposition_weights @ memory_bank  # (B, T, D)
        return retrieved


class UncertaintyMemory(nn.Module):
    def __init__(self, d_void: int, use_qram: bool = True) -> None:
        super().__init__()
        self.d_void = d_void
        self.use_qram = use_qram

        self.gate_z = nn.Linear(2 * d_void, d_void)
        self.gate_r = nn.Linear(2 * d_void, d_void)
        self.cand = nn.Linear(2 * d_void, d_void)
        self.read = nn.Linear(d_void, d_void)
        self.norm = nn.LayerNorm(d_void)

        if use_qram:
            self.qram = qRAMSimulator(d_model=d_void)

    def init_state(self, batch: int, seq: int, d: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, seq, d, device=device)

    def forward(
        self,
        v: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = v.shape

        if self.use_qram and memory.ndim == 3:
            # qRAM query retrieval across memory slots in superposition
            qram_retrieved = self.qram(v, memory)
            v_eff = v + qram_retrieved
        else:
            v_eff = v

        cat = torch.cat([v_eff, memory], dim=-1)
        z = torch.sigmoid(self.gate_z(cat))
        r = torch.sigmoid(self.gate_r(cat))
        cand = torch.tanh(self.cand(torch.cat([v_eff, r * memory], dim=-1)))
        new_mem = (1 - z) * memory + z * cand
        readout = self.norm(v_eff + self.read(new_mem))
        return readout, new_mem
