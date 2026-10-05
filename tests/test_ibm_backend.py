"""Unit tests for IBMQuantumBackend in voidformer/quantum/ibm_backend.py."""

from unittest.mock import patch
import pytest
import numpy as np
import torch

from voidformer.quantum.ibm_backend import (
    IBMBackend,
    build_circuit,
    save_account,
    get_qc_for_n_qubit_GHZ_state,
)
from voidformer.quantum_init import get_backend


def test_build_circuit():
    ops = [("h", 0), ("cx", 0, 1), ("rx", 0, 0.5)]
    qc = build_circuit(n_qubits=2, ops=ops, measure=True)
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2


def test_get_qc_for_n_qubit_GHZ_state():
    qc = get_qc_for_n_qubit_GHZ_state(4)
    assert qc.num_qubits == 4


def test_save_account():
    with patch("qiskit_ibm_runtime.QiskitRuntimeService.save_account") as mock_save:
        save_account(token="dummy_token_1234567890123456789012345678901234", instance="crn:v1:test")
        mock_save.assert_called_once_with(
            token="dummy_token_1234567890123456789012345678901234",
            channel="ibm_quantum_platform",
            overwrite=True,
            instance="crn:v1:test",
        )


def test_ibm_backend_simulator_counts():
    backend = IBMBackend(use_simulator=True, shots=500)
    bell_ops = [("h", 0), ("cx", 0, 1)]
    counts = backend.run(n_qubits=2, ops=bell_ops)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 500
    assert "00" in counts or "11" in counts


def test_ibm_backend_probabilities():
    backend = IBMBackend(use_simulator=True, shots=1000)
    bell_ops = [("h", 0), ("cx", 0, 1)]
    probs = backend.probabilities(n_qubits=2, ops=bell_ops)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 4
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0


def test_ibm_backend_expectation_values():
    backend = IBMBackend(use_simulator=True, shots=1000)
    bell_ops = [("h", 0), ("cx", 0, 1)]
    evs = backend.expectation_values(bell_ops, observables_labels=["ZZ", "IZ"], n_qubits=2)
    assert isinstance(evs, np.ndarray)
    assert len(evs) == 2
    assert pytest.approx(evs[0], abs=0.1) == 1.0


def test_get_backend_factory():
    sim_backend = get_backend("simulator")
    assert sim_backend is None

    aer_backend = get_backend("ibm_aer")
    assert isinstance(aer_backend, IBMBackend)
    assert aer_backend.use_simulator is True
