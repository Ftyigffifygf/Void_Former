"""Numerical Unit Tests for VoidFormer Quantum Core.

Asserts exact numerical outputs, quantum state vector fidelity against Qiskit,
concurrence metrics, hard collapse STE gradient flow, and cross-token entanglement.
"""

from __future__ import annotations

import math
import torch
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from voidformer.quantum import (
    QuantumStateVector,
    HadamardGate,
    CNOTGate,
    PauliXGate,
    ToffoliGate,
    BellStateGenerator,
    EntanglementManager,
    MeasurementLayer,
    CollapseProtocol,
)


def test_rotation_gates_numerical():
    """Verify RX, RY, RZ gate matrix unitarity and state rotation."""
    from voidformer.quantum import RXGate, RYGate, RZGate, QuantumCircuit, BackendIntegration

    for gate_cls in (RXGate, RYGate, RZGate):
        gate = gate_cls(theta=math.pi / 3)
        assert gate.verify_unitary()

    # Verify RZ(pi) on |0> gives e^(-i pi/2) |0>
    amps = torch.tensor([[[1.0 + 0j, 0.0 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=1)
    rz = RZGate(theta=math.pi)
    out_state = rz.apply(state, target_qubits=[0])
    expected = torch.tensor([[[-1j, 0j]]], dtype=torch.complex64)
    assert torch.allclose(out_state.amplitudes, expected, atol=1e-5)

    # Test IBM Quantum service authentication error handling when no token provided
    with pytest.raises((RuntimeError, ValueError, Exception)):
        BackendIntegration.get_ibm_service(token="invalid_test_token_123")

    # Test batch execution on Aer simulator
    qc1 = QuantumCircuit(name="batch_c1", n_qubits=2)
    qc1.add_rotation(0, "X", math.pi)
    qc2 = QuantumCircuit(name="batch_c2", n_qubits=2)
    qc2.add_rotation(1, "X", math.pi)
    batch_counts = BackendIntegration.execute_batch_on_aer([qc1, qc2], shots=100)
    assert len(batch_counts) == 2

    # Test poll_job mock handling
    class MockDoneJob:
        def status(self): return "DONE"
        def result(self): return {"0": 100}

    class MockErrorJob:
        def status(self): return "ERROR"

    class MockTimeoutJob:
        def status(self): return "RUNNING"
        def cancel(self): pass

    assert BackendIntegration.poll_job(MockDoneJob(), timeout_seconds=1.0) == {"0": 100}

    with pytest.raises(RuntimeError):
        BackendIntegration.poll_job(MockErrorJob(), timeout_seconds=1.0)

    with pytest.raises(TimeoutError):
        BackendIntegration.poll_job(MockTimeoutJob(), timeout_seconds=0.1, poll_interval=0.05)

    # Test circuit conversion to Qiskit with RX, RY, RZ
    qc = QuantumCircuit(name="rot_test", n_qubits=2)
    qc.add_rotation(0, "X", math.pi / 2)
    qc.add_rotation(1, "Y", math.pi / 4)
    qc.add_rotation(0, "Z", math.pi / 3)
    qiskit_qc = BackendIntegration.to_qiskit_circuit(qc)
    assert len(qiskit_qc.data) == 3


def test_hadamard_numerical():
    """Hadamard on |0> -> (|0> + |1>) / sqrt(2)."""
    amps = torch.tensor([[[1.0 + 0j, 0.0 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=1)
    H = HadamardGate()
    out = H.apply(state, target_qubits=[0])

    expected = torch.tensor([[[1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j]]], dtype=torch.complex64)
    assert torch.allclose(out.amplitudes, expected, atol=1e-5)


def test_bell_state_concurrence_exact():
    """After H then CNOT on |00>, concurrence == 1.0."""
    amps = torch.tensor([[[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=2)

    H = HadamardGate()
    CNOT = CNOTGate()

    s1 = H.apply(state, target_qubits=[0])
    s2 = CNOT.apply(s1, target_qubits=[0, 1])

    em = EntanglementManager(n_qubits_per_token=2, max_seq_len=16)
    c = em.compute_concurrence(s2)

    assert abs(c.item() - 1.0) < 1e-5, f"Expected concurrence 1.0, got {c.item()}"


def test_qiskit_statevector_comparison_phi_plus():
    """Compare Bell Phi+ statevector against Qiskit Statevector."""
    gen = BellStateGenerator()
    phi_plus = gen.create_bell_phi_plus(batch_size=1, seq_len=1).amplitudes[0, 0]

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qiskit_sv = Statevector.from_instruction(qc)

    qiskit_tensor = torch.tensor(qiskit_sv.data, dtype=torch.complex64)
    assert torch.allclose(phi_plus, qiskit_tensor, atol=1e-5)


def test_qiskit_statevector_comparison_toffoli():
    """Compare Toffoli circuit statevector against Qiskit Statevector."""
    # Circuit: X(0), X(1), Toffoli(0, 1, 2) on |000> -> |111> (index 7)
    amps = torch.zeros(1, 1, 8, dtype=torch.complex64)
    amps[0, 0, 0] = 1.0
    state = QuantumStateVector(amplitudes=amps, n_qubits=3)

    X = PauliXGate()
    Toff = ToffoliGate()

    s1 = X.apply(state, target_qubits=[0])
    s2 = X.apply(s1, target_qubits=[1])
    s3 = Toff.apply(s2, target_qubits=[0, 1, 2])

    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.mcx([0, 1], 2)
    qiskit_sv = Statevector.from_instruction(qc)

    s3_flat = s3.amplitudes[0, 0]
    qiskit_tensor = torch.tensor(qiskit_sv.data, dtype=torch.complex64)
    assert torch.allclose(s3_flat, qiskit_tensor, atol=1e-5)


def test_hard_collapse_ste_gradient_flow():
    """Verify gradients flow through hard collapse using Straight-Through Estimator."""
    amps = torch.tensor([[[0.6 + 0j, 0.8 + 0j, 0.0 + 0j, 0.0 + 0j]]], dtype=torch.complex64, requires_grad=True)
    state = QuantumStateVector(amplitudes=amps, n_qubits=2)
    layer = MeasurementLayer(n_qubits=2, d_output=4, collapse_protocol=CollapseProtocol.HARD)

    out, col_state, _ = layer(state, return_collapsed_state=True)
    loss = out.sum() + col_state.amplitudes.abs().sum()
    loss.backward()

    assert amps.grad is not None
    assert torch.any(amps.grad != 0), "Gradients failed to flow through hard collapse STE"


def test_cross_token_entanglement_concurrence():
    """Verify cross-token entangler creates non-zero entanglement."""
    amps = torch.zeros(1, 4, 4, dtype=torch.complex64)
    amps[0, :, 0] = 1.0  # All tokens in |00>
    state = QuantumStateVector(amplitudes=amps, n_qubits=2)

    em = EntanglementManager(n_qubits_per_token=2, max_seq_len=16)
    entangled_state, shared_reg = em.apply_cross_token_entanglement(state)

    is_entangled, measure = em.verify_entanglement(entangled_state)
    assert is_entangled.item() is True
    assert measure.mean().item() > 0.01


def test_quantum_voidformer_model_end_to_end():
    """Verify QuantumVoidFormer instantiates, runs forward/backward passes, and computes gradients."""
    from voidformer.models.quantum_voidformer import QuantumVoidFormer

    vocab_size = 64
    d_model = 32
    seq_len = 8
    batch_size = 2

    model = QuantumVoidFormer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        n_qubits_per_token=2,
        max_seq_len=seq_len,
        collapse_protocol="entropy_gated",
        enable_entanglement=True,
    )

    ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(ids, return_diagnostics=True)

    assert output.logits.shape == (batch_size, seq_len, vocab_size)
    assert torch.isfinite(output.logits).all()

    loss = output.logits.sum()
    loss.backward()

    # Check that model weights received gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and any(g > 0 for g in grad_norms)
