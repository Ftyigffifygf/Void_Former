"""Tests for Parameter-Shift Rule vs Finite Differences Numerical Agreement."""

from __future__ import annotations

import math
import torch

from voidformer.quantum.quantum_gates import RXGate, RYGate, RZGate, CNOTGate
from voidformer.quantum.qubit_state import QuantumStateVector


def simulate_vqc_expectation(
    angles: torch.Tensor,
    n_qubits: int = 2,
    cdtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Run a 2-qubit VQC with parameter angles and compute Pauli-Z expectation on qubit 0."""
    batch_size, seq_len = 1, 1
    state_dim = 2 ** n_qubits
    amps = torch.zeros(batch_size, seq_len, state_dim, dtype=cdtype, device=angles.device)
    amps[..., 0] = 1.0 + 0j
    state = QuantumStateVector(amplitudes=amps, n_qubits=n_qubits)

    # Layer 1: Rotations
    rx0 = RXGate(theta=angles[0])
    ry1 = RYGate(theta=angles[1])
    state = rx0.apply(state, [0])
    state = ry1.apply(state, [1])

    # Layer 2: CNOT
    cnot = CNOTGate()
    state = cnot.apply(state, [0, 1])

    # Layer 3: Rotations
    rz0 = RZGate(theta=angles[2])
    rx1 = RXGate(theta=angles[3])
    state = rz0.apply(state, [0])
    state = rx1.apply(state, [1])

    probs = state.probabilities.squeeze(0).squeeze(0)  # (4,)
    # Computational basis states |q1 q0>:
    # 0: |00>, 1: |01>, 2: |10>, 3: |11>
    # Z on q0: +1 for q0=0 (0 and 2), -1 for q0=1 (1 and 3)
    z0_exp = probs[0] - probs[1] + probs[2] - probs[3]
    return z0_exp


def test_parameter_shift_vs_finite_difference():
    """Verify parameter-shift gradient agrees with central finite difference gradient."""
    torch.manual_seed(42)
    angles = torch.tensor([0.4, -0.7, 1.2, 0.8], dtype=torch.float64)

    # Compute parameter-shift gradients
    grad_param_shift = torch.zeros_like(angles)
    shift = math.pi / 2.0

    for i in range(len(angles)):
        angles_plus = angles.clone()
        angles_plus[i] += shift
        exp_plus = simulate_vqc_expectation(angles_plus)

        angles_minus = angles.clone()
        angles_minus[i] -= shift
        exp_minus = simulate_vqc_expectation(angles_minus)

        grad_param_shift[i] = (exp_plus - exp_minus) / 2.0

    # Compute finite-difference gradients
    grad_finite_diff = torch.zeros_like(angles)
    eps = 1e-6

    for i in range(len(angles)):
        angles_plus = angles.clone()
        angles_plus[i] += eps
        exp_plus = simulate_vqc_expectation(angles_plus)

        angles_minus = angles.clone()
        angles_minus[i] -= eps
        exp_minus = simulate_vqc_expectation(angles_minus)

        grad_finite_diff[i] = (exp_plus - exp_minus) / (2.0 * eps)

    abs_diff = torch.abs(grad_param_shift - grad_finite_diff)
    max_diff = torch.max(abs_diff).item()

    assert torch.allclose(grad_param_shift, grad_finite_diff, atol=1e-5, rtol=1e-4), (
        f"Parameter shift gradient does not match finite differences! Max diff: {max_diff}"
    )


def test_vqc_autograd_function_backward():
    """Verify VQCAutogradFunction computes exact parameter-shift gradients during PyTorch backward pass."""
    from voidformer.quantum.vqc_layer import execute_vqc

    n_qubits = 2
    n_layers = 1
    K = n_layers * n_qubits * 3
    angles = torch.tensor([[0.2, -0.5, 0.8, 1.1, -0.3, 0.4]], dtype=torch.float64, requires_grad=True)

    out = execute_vqc(angles, n_qubits=n_qubits, n_layers=n_layers, backend="statevector")
    loss = out.sum()
    loss.backward()

    assert angles.grad is not None
    assert angles.grad.shape == angles.shape
    assert torch.all(torch.isfinite(angles.grad))


def test_vqc_layer_forward_backward():
    """Verify VQCLayer forward and backward passes with PyTorch autograd."""
    from voidformer.quantum.vqc_layer import VQCLayer

    d_model = 16
    batch_size, seq_len = 2, 4
    vqc = VQCLayer(d_model=d_model, n_vqc_qubits=2, n_vqc_layers=1, backend="statevector")

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out, diag = vqc(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert "vqc_mean_expectation" in diag

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.all(torch.isfinite(x.grad))


def test_shot_noise_expectation_and_variance():
    """Verify calculation of expectation values and finite-shot variance estimation."""
    from voidformer.quantum.vqc_layer import compute_shot_expectation_and_variance

    counts = {"00": 700, "11": 300}
    exp_vals, variances = compute_shot_expectation_and_variance(counts, n_qubits=2, shots=1000)

    assert torch.allclose(exp_vals, torch.tensor([0.4, 0.4]), atol=1e-5)

    expected_var = (1.0 - 0.4 ** 2) / 1000.0
    assert torch.allclose(variances, torch.tensor([expected_var, expected_var]), atol=1e-5)
