"""Audit and verification test suite for Q-INTELLIGENCE."""

import pytest
import torch
import numpy as np

from qai.transformer import ClassicalTransformer, HybridQuantumTransformer
from qai.quantum.circuit_builder import CircuitBuilder
from qai.core.device_manager import DeviceManager
from qai.hardware import IBMBackend, HardwareValidator, QuantumAuthenticator
from qai.datasets import get_xor_dataset, create_train_val_loaders
from qai.neural import QuantumANN
from qai.training import Trainer


def test_no_causal_leakage():
    """Verify strictly that future tokens do not influence past predictions."""
    t1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    t2 = torch.tensor([[1, 2, 3, 7]], dtype=torch.long)

    c_model = ClassicalTransformer(vocab_size=10, d_model=8, n_heads=1, max_seq_len=8)
    q_model = HybridQuantumTransformer(vocab_size=10, d_model=8, n_qubits=2, max_seq_len=8)

    c_model.eval()
    q_model.eval()

    with torch.no_grad():
        c_out1 = c_model(t1)
        c_out2 = c_model(t2)
        assert torch.allclose(c_out1[:, :3, :], c_out2[:, :3, :], atol=1e-6)

        q_out1 = q_model(t1)
        q_out2 = q_model(t2)
        assert torch.allclose(q_out1[:, :3, :], q_out2[:, :3, :], atol=1e-6)


def test_finite_difference_vs_autograd():
    """Checks that autograd gradients of quantum circuits match finite differences."""
    dev = DeviceManager.create_device("default.qubit", n_qubits=2)
    builder = CircuitBuilder(dev, n_qubits=2, n_layers=1, diff_method="backprop")
    qnode = builder.build_qnode()

    inputs = torch.tensor([0.4, 0.6])
    weights = torch.tensor([[[0.3, 0.5], [0.2, 0.4]]], requires_grad=True)

    out = torch.stack(qnode(inputs, weights)).sum()
    grad_auto = torch.autograd.grad(out, weights)[0]

    eps = 1e-4
    grad_fd = torch.zeros_like(weights)
    for i in range(weights.shape[1]):
        for j in range(weights.shape[2]):
            w_plus = weights.clone().detach()
            w_minus = weights.clone().detach()
            w_plus[0, i, j] += eps
            w_minus[0, i, j] -= eps
            p_plus = torch.stack(qnode(inputs, w_plus)).sum()
            p_minus = torch.stack(qnode(inputs, w_minus)).sum()
            grad_fd[0, i, j] = (p_plus - p_minus) / (2 * eps)

    diff = torch.norm(grad_auto - grad_fd).item()
    assert diff < 1e-3, f"Gradient mismatch: {diff}"


def test_hardware_safeguards_raise_errors():
    """Confirms hardware execution cannot proceed or fall back silently without authorization."""
    backend = IBMBackend(allow_hardware_execution=False)
    with pytest.raises(PermissionError):
        backend.get_device(2)

    status = HardwareValidator.check_hardware_status(allow_hardware=False)
    assert status["circuit_execution"] in ["NOT AVAILABLE", "NOT AUTHORIZED"]
