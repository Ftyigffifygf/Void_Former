"""Dual Embedding System with State-Superposition Tokens Generator (|S_c>, |S_v>).

Every token simultaneously produces:
    - classical embedding   E_c(token)   — deterministic semantic anchor
    - void embedding        E_v(token)   — latent ambiguity / future possibility

The two spaces are generated in quantum superposition |T> = alpha |S_c> + beta |S_v> + gamma I(|S_c>, |S_v>)
with entropy-driven Fisher metric tensor geometry.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from voidformer.quantum.state_superposition import StateSuperpositionTokenGenerator


class DualEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_void: int,
        max_seq_len: int,
        coupling_init: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_void = d_void

        self.classical_emb = nn.Embedding(vocab_size, d_model)
        self.void_emb = nn.Embedding(vocab_size, d_void)

        self.pos_classical = nn.Embedding(max_seq_len, d_model)
        self.pos_void = nn.Embedding(max_seq_len, d_void)

        self.couple_c2v = nn.Linear(d_model, d_void, bias=False)
        self.couple_v2c = nn.Linear(d_void, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(coupling_init)))

        self.superposition_gen = StateSuperpositionTokenGenerator(d_model=d_model, n_qubits=4)

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.classical_emb.weight, std=0.02)
        nn.init.normal_(self.void_emb.weight, std=0.02)
        nn.init.normal_(self.pos_classical.weight, std=0.01)
        nn.init.normal_(self.pos_void.weight, std=0.01)
        nn.init.normal_(self.couple_c2v.weight, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.couple_v2c.weight, std=1.0 / math.sqrt(self.d_void))

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ids: (B, T) -> (classical (B,T,d_model), void (B,T,d_void))"""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)

        c = self.classical_emb(ids) + self.pos_classical(pos)
        v = self.void_emb(ids) + self.pos_void(pos)

        # Generate state superposition |T>
        c_sup, metric, diag = self.superposition_gen(c)

        g = torch.tanh(self.gate)
        c2v = self.couple_c2v(c_sup)
        v2c = self.couple_v2c(v)

        c_out = c_sup + g * v2c
        v_out = v + g * c2v

        return self.drop(c_out), self.drop(v_out)
