"""Unit tests for Quantum Superposition Reasoning Engine (QSRE)."""

from __future__ import annotations

import torch

from voidformer.quantum.superposition_thinking import (
    QuantumHilbertMemory,
    UnitaryThinkingLoop,
    QuantumAmplitudeOracle,
    SuperposedBornDecoder,
    QuantumSuperpositionReasoningEngine,
)
from voidformer.models.quantum_voidformer import QuantumVoidFormer


def test_quantum_hilbert_memory():
    B, T, D, Q = 2, 8, 64, 4
    memory = QuantumHilbertMemory(d_model=D, n_qubits=Q)
    x = torch.randn(B, T, D)
    psi = memory(x)

    # Check output shape
    assert psi.shape == (B, T, 2 ** Q)
    # Check complex dtype
    assert psi.is_complex()

    # Check state vector normalization \sum |\alpha_i|^2 = 1
    norms = torch.norm(psi, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_unitary_thinking_loop():
    B, T, Q = 2, 8, 4
    hilbert_dim = 2 ** Q
    loop = UnitaryThinkingLoop(n_qubits=Q, thinking_steps=4)
    assert loop.phase_weights.shape == (4, Q, 3)

    psi_in = torch.randn(B, T, hilbert_dim, dtype=torch.complex64)
    norm_before = torch.norm(psi_in, dim=-1)

    psi_out = loop(psi_in, step=0)
    assert psi_out.shape == (B, T, hilbert_dim)
    assert psi_out.is_complex()

    # Phase rotation should preserve state vector norm
    norm_after = torch.norm(psi_out, dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


def test_quantum_amplitude_oracle():
    B, T, Q = 2, 8, 4
    hilbert_dim = 2 ** Q
    oracle = QuantumAmplitudeOracle(n_qubits=Q)

    psi_raw = torch.randn(B, T, hilbert_dim, dtype=torch.complex64)
    psi_in = psi_raw / (torch.norm(psi_raw, dim=-1, keepdim=True) + 1e-8)

    psi_out = oracle(psi_in)
    assert psi_out.shape == (B, T, hilbert_dim)
    assert psi_out.is_complex()

    # Re-normalization check
    norms = torch.norm(psi_out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_superposed_born_decoder():
    B, T, D, Q = 2, 8, 64, 4
    hilbert_dim = 2 ** Q
    decoder = SuperposedBornDecoder(d_model=D, n_qubits=Q)

    psi = torch.randn(B, T, hilbert_dim, dtype=torch.complex64)
    out = decoder(psi)

    assert out.shape == (B, T, D)
    assert not out.is_complex()


def test_qsre_full_module_forward_and_backward():
    B, T, D, Q, K = 2, 8, 64, 4, 4
    qsre = QuantumSuperpositionReasoningEngine(
        d_model=D, n_vqc_qubits=Q, thinking_steps=K
    )

    x = torch.randn(B, T, D, requires_grad=True)
    out = qsre(x)

    assert out.shape == (B, T, D)
    assert not out.is_complex()

    # Backward pass / gradient propagation
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (B, T, D)
    assert qsre.thinking_loop.phase_weights.grad is not None
    assert qsre.oracle.oracle_proj.weight.grad is not None


def test_qsre_quantum_voidformer_integration():
    model = QuantumVoidFormer(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        n_qubits_per_token=4,
        n_vqc_qubits=4,
        use_superposition_thinking=True,
        thinking_steps=4,
        max_seq_len=64,
    )

    ids = torch.randint(0, 100, (2, 8))
    output = model(ids, return_diagnostics=True)

    assert output.logits.shape == (2, 8, 100)
    assert len(output.layer_diagnostics) == 2
    for layer_diag in output.layer_diagnostics:
        assert layer_diag.get("superposition_thinking") is True
