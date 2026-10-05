"""State-Superposition Tokens Generator (|S_c>, |S_v>), Entropy-Driven Fisher Metric Tensors, and Quantum Tokenizer."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from voidformer.quantum.qubit_state import QuantumStateVector


class QuantumTokenizer(nn.Module):
    """Quantum Tokenizer bridging classical data into Hilbert space qubits."""

    def __init__(self, vocab_size: int, d_model: int, n_qubits: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.hilbert_dim = 2 ** n_qubits

        self.classical_emb = nn.Embedding(vocab_size, d_model)
        self.quantum_proj = nn.Linear(d_model, 2 * self.hilbert_dim)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, QuantumStateVector]:
        x = self.classical_emb(ids)
        flat = self.quantum_proj(x)
        real, imag = flat.chunk(2, dim=-1)
        amps = torch.complex(real, imag)
        norm = torch.norm(amps, dim=-1, keepdim=True).clamp_min(1e-12)
        norm_amps = amps / norm

        state = QuantumStateVector(amplitudes=norm_amps, n_qubits=self.n_qubits)
        return x, state


class EntropyDrivenFisherMetricTensor(nn.Module):
    """Quantum Fisher Information Metric (QFIM) tensor driven by state entropy."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def compute_fisher_metric(
        self, state_amps: torch.Tensor, entropy: torch.Tensor
    ) -> torch.Tensor:
        B, T, dim = state_amps.shape
        state_norm = F.normalize(state_amps.real, p=2, dim=-1)
        metric = torch.matmul(state_norm.unsqueeze(-1), state_norm.unsqueeze(-2))
        entropy_weight = (1.0 + entropy).unsqueeze(-1).unsqueeze(-1)
        return metric * entropy_weight


class StateSuperpositionTokenGenerator(nn.Module):
    """Superposition Token Generator producing |S_c> (certainty) and |S_v> (void) quantum state superpositions."""

    def __init__(self, d_model: int, n_qubits: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits

        self.proj_certainty = nn.Linear(d_model, d_model)
        self.proj_void = nn.Linear(d_model, d_model)
        self.proj_interference = nn.Linear(d_model, d_model)

        self.tokenizer = QuantumTokenizer(vocab_size=256, d_model=d_model, n_qubits=n_qubits)
        self.fisher_metric = EntropyDrivenFisherMetricTensor(d_model=d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        S_c = self.proj_certainty(x)
        S_v = self.proj_void(x)

        interference = torch.tanh(self.proj_interference(S_c * S_v))

        alpha = torch.sigmoid(torch.mean(S_c, dim=-1, keepdim=True))
        beta = torch.sigmoid(torch.mean(S_v, dim=-1, keepdim=True))
        gamma = 1.0 - alpha - beta

        T_state = alpha * S_c + beta * S_v + gamma * interference

        probs = F.softmax(T_state, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-9)), dim=-1)

        metric = self.fisher_metric.compute_fisher_metric(T_state, entropy)

        diagnostics = {
            "alpha_mean": alpha.mean().item(),
            "beta_mean": beta.mean().item(),
            "gamma_mean": gamma.mean().item(),
            "fisher_metric_norm": metric.norm().item(),
        }

        return T_state, metric, diagnostics
