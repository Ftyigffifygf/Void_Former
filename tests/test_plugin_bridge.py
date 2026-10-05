"""Unit tests for Universal AI Plugin Bridge & Quantum Personal Space Data Vault."""

from __future__ import annotations

import torch
import torch.nn as nn

from voidformer.quantum.plugin_bridge import (
    QuantumPersonalSpaceVault,
    QuantumVoidFormerAIPlugin,
)
from voidformer.harness.quantum_bridge import QuantumEngineeringBridge


def test_quantum_personal_space_vault():
    B, T, D, Q = 2, 8, 64, 4
    vault = QuantumPersonalSpaceVault(d_model=D, n_vqc_qubits=Q)

    x = torch.randn(B, T, D)
    x_prot, psi_prot = vault.protect_data(x)

    assert x_prot.shape == (B, T, D)
    assert psi_prot.shape == (B, T, 2 ** Q)
    assert psi_prot.is_complex()

    # Norm of protected quantum state = 1
    norms = torch.norm(psi_prot, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_quantum_voidformer_ai_plugin_standalone():
    B, T, D, Q = 2, 8, 64, 4
    plugin = QuantumVoidFormerAIPlugin(
        base_ai_model=None,
        d_model=D,
        n_vqc_qubits=Q,
        thinking_steps=2,
        use_quantum_moe=True,
        enable_data_vault=True,
    )

    x = torch.randn(B, T, D, requires_grad=True)
    out = plugin(x)

    assert out.shape == (B, T, D)

    # Test qubit space rotation
    x_rot = plugin.rotate_qubit_space(x, angle_x=0.1, angle_z=0.2)
    assert x_rot.shape == (B, T, D)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None


def test_quantum_voidformer_ai_plugin_wrapping_external_ai_model():
    # External PyTorch / HuggingFace style dummy AI block
    base_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 64),
    )

    plugin = QuantumVoidFormerAIPlugin(
        base_ai_model=base_model,
        d_model=64,
        n_vqc_qubits=4,
        thinking_steps=2,
        enable_data_vault=True,
    )

    x = torch.randn(2, 8, 64, requires_grad=True)
    out = plugin(x)

    assert out.shape == (2, 8, 64)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None


def test_quantum_engineering_bridge_auto_fallback():
    bridge = QuantumEngineeringBridge(backend_type="auto")

    assert bridge.active_backend == "virtual_statevector"
    assert bridge.has_physical_qpu is False

    res = bridge.run_circuit(None, shots=100)
    assert res["backend"] == "virtual_statevector"
    assert res["physical_qpu_used"] is False

    plugin = bridge.create_ai_plugin(d_model=64, n_vqc_qubits=4)
    x = torch.randn(2, 8, 64)
    out = plugin(x)
    assert out.shape == (2, 8, 64)
