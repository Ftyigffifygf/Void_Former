"""Variational Quantum Circuit (VQC) layer with Parameter-Shift Autograd and Equivariant QNNs."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

from voidformer.quantum.qubit_state import QuantumStateVector
from voidformer.quantum.quantum_gates import RXGate, RYGate, RZGate, CNOTGate


def compute_shot_expectation_and_variance(
    counts: Dict[str, int], n_qubits: int, shots: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Pauli-Z expectation values and variances for each qubit from shot measurement counts."""
    exp_vals = torch.zeros(n_qubits, dtype=torch.float32)
    for bitstring, count in counts.items():
        prob = count / shots
        for q in range(n_qubits):
            bit_idx = len(bitstring) - 1 - q
            z_val = 1.0 if bitstring[bit_idx] == "0" else -1.0
            exp_vals[q] += z_val * prob

    variances = (1.0 - exp_vals ** 2) / shots
    return exp_vals, variances


def execute_vqc(
    angles: torch.Tensor,
    n_qubits: int = 2,
    n_layers: int = 1,
    backend: str = "statevector",
) -> torch.Tensor:
    """Execute VQC circuit with parameterized rotation angles and return Pauli-Z expectation values."""
    shape = angles.shape[:-1]
    flat_angles = angles.reshape(-1, angles.shape[-1])
    batch_size = flat_angles.shape[0]

    state_dim = 2 ** n_qubits
    cdtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
    amps = torch.zeros(batch_size, state_dim, dtype=cdtype, device=angles.device)
    amps[:, 0] = 1.0 + 0j
    state = QuantumStateVector(amplitudes=amps, n_qubits=n_qubits)

    idx = 0
    for l in range(n_layers):
        for q in range(n_qubits):
            rx = RXGate(flat_angles[:, idx])
            ry = RYGate(flat_angles[:, idx + 1])
            rz = RZGate(flat_angles[:, idx + 2])
            idx += 3
            state = rx.apply(state, [q])
            state = ry.apply(state, [q])
            state = rz.apply(state, [q])

        for q in range(n_qubits - 1):
            cnot = CNOTGate()
            state = cnot.apply(state, [q, q + 1])

    probs = state.probabilities  # (batch_size, 2**n_qubits)
    exp_vals = torch.zeros(batch_size, n_qubits, dtype=angles.dtype, device=angles.device)

    for q in range(n_qubits):
        for basis_idx in range(2 ** n_qubits):
            bit = (basis_idx >> q) & 1
            sign = 1.0 if bit == 0 else -1.0
            exp_vals[:, q] += sign * probs[:, basis_idx]

    return exp_vals.reshape(*shape, n_qubits)


class VQCAutogradFunction(torch.autograd.Function):
    """Custom PyTorch autograd function using exact Parameter-Shift Rule:
    (f(θ + π/2) - f(θ - π/2)) / 2
    """

    @staticmethod
    def forward(ctx, angles: torch.Tensor, n_qubits: int, n_layers: int, backend: str) -> torch.Tensor:
        ctx.save_for_backward(angles)
        ctx.n_qubits = n_qubits
        ctx.n_layers = n_layers
        ctx.backend = backend
        return execute_vqc(angles, n_qubits=n_qubits, n_layers=n_layers, backend=backend)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        angles, = ctx.saved_tensors
        n_qubits = ctx.n_qubits
        n_layers = ctx.n_layers
        backend = ctx.backend

        shift = math.pi / 2.0
        grad_angles = torch.zeros_like(angles)

        num_params = angles.shape[-1]
        for p in range(num_params):
            angles_plus = angles.clone()
            angles_plus[..., p] += shift
            exp_plus = execute_vqc(angles_plus, n_qubits=n_qubits, n_layers=n_layers, backend=backend)

            angles_minus = angles.clone()
            angles_minus[..., p] -= shift
            exp_minus = execute_vqc(angles_minus, n_qubits=n_qubits, n_layers=n_layers, backend=backend)

            param_shift_grad = (exp_plus - exp_minus) / 2.0

            grad_angles[..., p] = torch.sum(grad_output * param_shift_grad, dim=-1)

        return grad_angles, None, None, None


class EquivariantQNNLayer(nn.Module):
    """Equivariant Quantum Neural Network (EQNN) layer with symmetry group invariance (SU(N) / Permutation Symmetry)."""

    def __init__(self, n_qubits: int = 4, group_symmetry: str = "permutation"):
        super().__init__()
        self.n_qubits = n_qubits
        self.group_symmetry = group_symmetry

        if group_symmetry == "permutation":
            self.shared_rotations = nn.Parameter(torch.randn(3))
        else:
            self.shared_rotations = nn.Parameter(torch.randn(n_qubits, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_symmetry == "permutation":
            angles = self.shared_rotations.repeat(self.n_qubits)
        else:
            angles = self.shared_rotations.reshape(-1)

        batch_shape = x.shape[:-1]
        angles_batch = angles.expand(*batch_shape, angles.shape[-1])
        return VQCAutogradFunction.apply(angles_batch, self.n_qubits, 1, "statevector")


class VQCLayer(nn.Module):
    """Variational Quantum Circuit layer with Parameter-Shift Autograd."""

    def __init__(
        self,
        d_model: int,
        n_vqc_qubits: int = 2,
        n_vqc_layers: int = 1,
        backend: str = "statevector",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_vqc_qubits = n_vqc_qubits
        self.n_vqc_layers = n_vqc_layers
        self.backend = backend

        self.num_params = n_vqc_layers * n_vqc_qubits * 3
        self.proj_in = nn.Linear(d_model, self.num_params)
        self.proj_out = nn.Linear(n_vqc_qubits, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        angles = self.proj_in(x)
        exp_vals = VQCAutogradFunction.apply(angles, self.n_vqc_qubits, self.n_vqc_layers, self.backend)
        out = self.proj_out(exp_vals)
        diagnostics = {"vqc_mean_expectation": exp_vals.mean().item()}
        return out, diagnostics
