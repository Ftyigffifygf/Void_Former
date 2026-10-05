"""Quantum State Vector representation for simulation."""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantumStateVector:
    """Quantum state vector supporting batched amplitudes."""

    def __init__(self, amplitudes: torch.Tensor, n_qubits: int):
        self.n_qubits = n_qubits
        self.amplitudes = amplitudes  # Shape: (..., 2**n_qubits)

    @property
    def probabilities(self) -> torch.Tensor:
        """Compute state probabilities |psi|^2."""
        return torch.abs(self.amplitudes) ** 2

    def density_matrix(self) -> torch.Tensor:
        """Compute state density matrix rho = |psi><psi| for single state or batch."""
        # amplitudes shape: (..., dim)
        amps = self.amplitudes.unsqueeze(-1)  # (..., dim, 1)
        amps_dagger = self.amplitudes.unsqueeze(-2).conj()  # (..., 1, dim)
        return torch.matmul(amps, amps_dagger)
