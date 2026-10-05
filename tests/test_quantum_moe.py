"""Unit tests for Quantum Superposition Mixture of Experts (MoE) & Tokenization."""

from __future__ import annotations

import torch

from voidformer.quantum.superposition_moe import (
    QuantumSuperpositionTokenEmbedder,
    QuantumSuperpositionExpert,
    QuantumSuperpositionRouter,
    QuantumSuperpositionMoE,
)
from voidformer.models.quantum_voidformer import QuantumVoidFormer
from voidformer.harness.model_factory import create_model


def test_quantum_superposition_token_embedder():
    B, T, V, D, Q = 2, 8, 100, 64, 4
    embedder = QuantumSuperpositionTokenEmbedder(vocab_size=V, d_model=D, n_vqc_qubits=Q)

    ids = torch.randint(0, V, (B, T))
    x_classical, psi = embedder(ids)

    assert x_classical.shape == (B, T, D)
    assert psi.shape == (B, T, 2 ** Q)
    assert psi.is_complex()

    norms = torch.norm(psi, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_quantum_superposition_expert():
    B, T, D, Q = 2, 8, 64, 4
    expert = QuantumSuperpositionExpert(d_model=D, n_vqc_qubits=Q, thinking_steps=2)

    x = torch.randn(B, T, D)
    out = expert(x)

    assert out.shape == (B, T, D)


def test_quantum_superposition_router():
    B, T, D, Q, E, K = 2, 8, 64, 4, 4, 2
    router = QuantumSuperpositionRouter(d_model=D, num_experts=E, n_vqc_qubits=Q)

    x = torch.randn(B, T, D)
    weights, indices = router(x, top_k=K)

    assert weights.shape == (B, T, K)
    assert indices.shape == (B, T, K)

    # Softmax sums to 1
    sum_weights = weights.sum(dim=-1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5)


def test_quantum_superposition_moe_full():
    B, T, D, Q, E, K = 2, 8, 64, 4, 4, 2
    moe = QuantumSuperpositionMoE(
        d_model=D, n_vqc_qubits=Q, num_experts=E, top_k_experts=K, thinking_steps=2
    )

    x = torch.randn(B, T, D, requires_grad=True)
    out = moe(x)

    assert out.shape == (B, T, D)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (B, T, D)


def test_quantum_voidformer_with_moe_and_quantum_embedder():
    model = QuantumVoidFormer(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        n_qubits_per_token=4,
        n_vqc_qubits=4,
        use_superposition_thinking=True,
        thinking_steps=2,
        use_quantum_moe=True,
        num_experts=4,
        top_k_experts=2,
        use_quantum_token_embedder=True,
        max_seq_len=64,
    )

    ids = torch.randint(0, 100, (2, 8))
    output = model(ids, return_diagnostics=True)

    assert output.logits.shape == (2, 8, 100)
    assert len(output.layer_diagnostics) == 2
    for layer_diag in output.layer_diagnostics:
        assert layer_diag.get("quantum_moe") is True


def test_harness_model_factory_moe():
    model = create_model(
        model_type="quantum",
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=64,
        use_quantum_moe=True,
        num_experts=4,
        top_k_experts=2,
        use_quantum_token_embedder=True,
    )

    ids = torch.randint(0, 100, (2, 8))
    output = model(ids)

    assert output.logits.shape == (2, 8, 100)
