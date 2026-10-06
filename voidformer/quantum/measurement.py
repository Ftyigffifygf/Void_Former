"""Classical Shadows State Tomography & Randomized Pauli Measurements."""

from __future__ import annotations

import torch


class ClassicalShadowsTomography:
    """Randomized Pauli Measurement protocol (Classical Shadows) for state tomography in O(log N) measurements."""

    def __init__(self, n_qubits: int, num_shadow_samples: int = 100):
        self.n_qubits = n_qubits
        self.num_shadow_samples = num_shadow_samples

    def sample_random_pauli_bases(self) -> torch.Tensor:
        """Return random Pauli bases (0=X, 1=Y, 2=Z), shape (num_shadow_samples, n_qubits)."""
        return torch.randint(0, 3, (self.num_shadow_samples, self.n_qubits))

    def reconstruct_classical_shadow(
        self,
        pauli_bases: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct approximate density matrix rho_hat from randomized Pauli measurement outcomes.

        pauli_bases: (num_samples, n_qubits), values 0=X, 1=Y, 2=Z
        outcomes:    (num_samples, n_qubits), values 0 or 1
        """
        expected = (pauli_bases.shape[0], self.n_qubits)
        if pauli_bases.dim() != 2 or pauli_bases.shape[1] != self.n_qubits:
            raise ValueError(
                f"pauli_bases must have shape (num_samples, {self.n_qubits}), "
                f"got {tuple(pauli_bases.shape)}"
            )
        if outcomes.shape != pauli_bases.shape:
            raise ValueError(
                f"outcomes shape {tuple(outcomes.shape)} must match "
                f"pauli_bases shape {tuple(pauli_bases.shape)}"
            )

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

                # Single-qubit inverse channel: 3|psi><psi| - I
                qubit_snap = 3.0 * proj - eye2
                snapshot = torch.kron(snapshot, qubit_snap)

            rho_hat_sum += snapshot

        return rho_hat_sum / num_samples

    def estimate_observable(
        self, rho_hat: torch.Tensor, observable: torch.Tensor
    ) -> float:
        """Estimate <O> = Tr(O rho_hat)."""
        return torch.trace(torch.matmul(observable.to(rho_hat.dtype), rho_hat)).real.item()
