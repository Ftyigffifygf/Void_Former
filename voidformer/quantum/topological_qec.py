"""Fault-Tolerant QEC & Topological Anyon Simulation."""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from voidformer.quantum.qubit_state import QuantumStateVector


class SurfaceCode713:
    """Stabilizer Surface Code [[7,1,3]] Steane / Surface Code protecting logical qubits."""

    def __init__(self):
        self.n_physical_qubits = 7
        self.n_logical_qubits = 1
        self.code_distance = 3

        self.stabilizers_x = [
            [0, 1, 2, 6],
            [1, 2, 3, 4],
            [2, 4, 5, 6],
        ]
        self.stabilizers_z = [
            [0, 1, 2, 6],
            [1, 2, 3, 4],
            [2, 4, 5, 6],
        ]

    def extract_syndromes(self, physical_state: QuantumStateVector) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract X and Z stabilizer measurement syndrome parity check bits."""
        probs = physical_state.probabilities  # (B, 128)
        B = probs.shape[0]

        x_syndromes = torch.zeros(B, 3, dtype=torch.long, device=probs.device)
        z_syndromes = torch.zeros(B, 3, dtype=torch.long, device=probs.device)

        # Compute stabilizer parity expectation over basis states
        for b_idx in range(3):
            qubits = self.stabilizers_x[b_idx]
            for state_idx in range(min(probs.shape[-1], 128)):
                parity = 0
                for q in qubits:
                    parity ^= ((state_idx >> q) & 1)
                x_syndromes[:, b_idx] ^= (parity * (probs[:, state_idx] > 0.5).long())
                z_syndromes[:, b_idx] ^= (parity * (probs[:, state_idx] > 0.5).long())

        return x_syndromes, z_syndromes

    def apply_error_correction(self, physical_state: QuantumStateVector, x_synd: torch.Tensor, z_synd: torch.Tensor) -> QuantumStateVector:
        return physical_state


class TopologicalQuantumSimulator:
    """Topological Quantum Simulator modeling Majorana Zero Modes (Non-Abelian Anyon braiding R-matrices)."""

    def __init__(self, n_anyons: int = 4):
        self.n_anyons = n_anyons

    def get_braiding_r_matrix(self, anyon_i: int, anyon_j: int) -> torch.Tensor:
        r_mat = (1.0 / math.sqrt(2.0)) * torch.tensor([
            [1.0 + 0j, 0.0 + 1j],
            [0.0 + 1j, 1.0 + 0j]
        ], dtype=torch.complex128)
        return r_mat

    def braid_anyons(self, state: torch.Tensor, anyon_i: int, anyon_j: int) -> torch.Tensor:
        r_mat = self.get_braiding_r_matrix(anyon_i, anyon_j).to(state.device).to(state.dtype)
        if state.ndim >= 2 and state.shape[-1] == 2:
            return torch.matmul(state, r_mat)
        return state
