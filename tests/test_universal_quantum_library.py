"""Unit tests for Universal Quantum SDK Registry, Quantum Circuit Library, and Self-Improving Engine."""

import pytest
import torch

from voidformer.quantum import (
    UniversalQuantumSDKRegistry,
    QuantumCircuitLibrary,
    QuantumSelfImprovingEngine,
)
from voidformer.models.quantum_voidformer import QuantumVoidFormer


def test_universal_sdk_registry_statuses():
    statuses = UniversalQuantumSDKRegistry.get_all_statuses()
    assert len(statuses) >= 15
    qiskit_status = [s for s in statuses if s["sdk_name"].startswith("Qiskit")][0]
    assert qiskit_status["installed"] is True


def test_quantum_circuit_library_circuits():
    bell = QuantumCircuitLibrary.bell_state()
    assert bell.n_qubits == 2
    assert bell.depth() == 2

    shor = QuantumCircuitLibrary.shor_9_qubit_code()
    assert shor.n_qubits == 9

    grover = QuantumCircuitLibrary.grover_search(n_qubits=2)
    assert grover.n_qubits == 2

    qft = QuantumCircuitLibrary.qft(n_qubits=3)
    assert qft.n_qubits == 3


def test_self_improving_engine():
    engine = QuantumSelfImprovingEngine(d_model=16, n_qubits=4)
    x = torch.randn(2, 4, 16)
    out, diag = engine(x, fidelity_score=0.98)
    assert out.shape == x.shape
    assert diag["adaptation_count"] == 1
    assert diag["self_improving_engine_active"] is True


def test_quantum_voidformer_integration_with_self_improving():
    model = QuantumVoidFormer(
        vocab_size=50,
        d_model=16,
        n_layers=1,
    )
    ids = torch.randint(0, 50, (2, 6))
    out = model(ids, return_diagnostics=True)
    assert out.logits.shape == (2, 6, 50)
    assert "self_improving_engine" in out.layer_diagnostics[0]
