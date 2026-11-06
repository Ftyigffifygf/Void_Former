"""Hybrid Quantum-Classical Neural Network."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
from qai.neural.quantum_ann import QuantumANN


class HybridANN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network:
    Combines classical dense feature extraction with a variational quantum core and classical readout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 16,
        out_features: int = 2,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend_name: str = "default.qubit",
        task: str = "classification",
        shots: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.classical_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, n_qubits),
        )
        self.quantum_core = QuantumANN(
            in_features=n_qubits,
            out_features=hidden_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            backend_name=backend_name,
            shots=shots,
        )
        self.classical_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.classical_encoder(x)
        q_feat = self.quantum_core(feat)
        out = self.classical_decoder(q_feat)
        if self.task == "regression" and out.shape[-1] == 1:
            return out.squeeze(-1)
        return out
