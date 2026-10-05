r"""Classical Shadows State Tomography & Randomized Pauli Measurements."""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from voidformer.quantum.qubit_state import QuantumStateVector


class ClassicalShadowsTomography:
    """Randomized Pauli Measurement protocol (Classical Shadows) for state tomography in O(log N) measurements."""

    def __init__(self, n_qubits: int, num_shadow_samples: int = 100):
        self.n_qubits = n_qubits
        self.num_shadow_samples = num_shadow_samples

    def sample_random_pauli_bases(self, batch_size: int = 1) -> torch.Tensor:
        return torch.randint(0, 3, (batch_size, self.num_shadow_samples, self.n_qubits))

    def reconstruct_classical_shadow(
        self,
        pauli_bases: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        r"""Reconstruct approximate density matrix rho_hat from randomized Pauli measurement outcomes."""
        num_samples = pauli_bases.shape[0]
        dim = 2 ** self.n_qubits
        dev = pauli_bases.device
        rho_hat_sum = torch.zeros(dim, dim, dtype=torch.complex128, device=dev)

        proj_x0 = 0.5 * torch.tensor([[1, 1], [1, 1]], dtype=torch.complex128, device=dev)
        proj_x1 = 0.5 * torch.tensor([[1, -1], [-1, 1]], dtype=torch.complex128, device=dev)

        proj_y0 = 0.5 * torch.tensor([[1, -1j], [1j, 1]], dtype=torch.complex128, device=dev)
        proj_y1 = 0.5 * torch.tensor([[1, 1j], [-1j, 1]], dtype=torch.complex128, device=dev)

        proj_z0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128, device=dev)
        proj_z1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128, device=dev)

        eye2 = torch.eye(2, dtype=torch.complex128, device=dev)

        for m in range(num_samples):
            snapshot = torch.tensor([[1.0 + 0j]], dtype=torch.complex128, device=dev)
            for q in range(self.n_qubits):
                b = pauli_bases[m, q].item()
                s = outcomes[m, q].item()

                if b == 0:
                    proj = proj_x0 if s == 0 else proj_x1
                elif b == 1:
                    proj = proj_y0 if s == 0 else proj_y1
                else:
                    proj = proj_z0 if s == 0 else proj_z1

                qubit_snap = 3.0 * proj - eye2
                snapshot = torch.kron(snapshot, qubit_snap)

            rho_hat_sum += snapshot

        rho_hat = rho_hat_sum / num_samples
        return rho_hat

    def estimate_observable(
        self, rho_hat: torch.Tensor, observable: torch.Tensor
    ) -> float:
        exp_val = torch.trace(torch.matmul(observable.to(rho_hat.dtype), rho_hat)).real.item()
        return exp_val
