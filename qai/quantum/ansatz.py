"""Parameterized Quantum Circuit (PQC) Ansatz architectures."""

from __future__ import annotations
import math
import torch
import pennylane as qml
from typing import Sequence, Optional
from qai.quantum.entanglement import EntanglementBuilder


class VariationalAnsatz:
    """Implements hardware-efficient, strongly-entangling, and identity-initialized ansatzes."""

    @staticmethod
    def hardware_efficient(
        weights: torch.Tensor,
        wires: Sequence[int],
        entanglement: str = "ring",
        rotations: Sequence[str] = ("RY", "RZ"),
    ) -> None:
        """
        Hardware-efficient ansatz:
        For each layer:
          Single qubit rotations (e.g. RY, RZ)
          Entangling CNOT gates
        """
        n_layers = weights.shape[0]
        n_qubits = len(wires)

        for layer in range(n_layers):
            # Single-qubit rotations
            for q in range(n_qubits):
                # weights shape: (n_layers, n_qubits, n_rotations)
                if weights.dim() >= 3 and weights.shape[-1] >= 2:
                    qml.RY(weights[layer, q, 0], wires=wires[q])
                    qml.RZ(weights[layer, q, 1], wires=wires[q])
                else:
                    qml.RY(weights[layer, q], wires=wires[q])

            # Entanglement layer
            if entanglement == "ring":
                EntanglementBuilder.ring_entanglement(wires)
            else:
                EntanglementBuilder.linear_entanglement(wires)

    @staticmethod
    def identity_near_init(n_layers: int, n_qubits: int, n_rotations: int = 2, scale: float = 0.01) -> torch.Tensor:
        """Initializes ansatz parameters near identity to prevent barren plateaus and vanishing gradients."""
        return torch.randn(n_layers, n_qubits, n_rotations) * scale
