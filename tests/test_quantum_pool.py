"""Tests for QuantumPool and mathematical foundation equations."""

import pytest
import torch
from qai.quantum.quantum_pool import (
    QuantumPool,
    QuantumTokenSuperposition,
    QuantumInterferenceLayer,
    StochasticQuantumAttention,
    QuantumCollapseEngine,
    QuantumUncertaintyMemory,
    RiemannianGeodesicEngine,
)


def test_quantum_token_superposition():
    sup = QuantumTokenSuperposition(d_model=8)
    sc = torch.randn(2, 3, 8)
    sv = torch.randn(2, 3, 8)
    t, a, b = sup(sc, sv)
    assert t.shape == torch.Size([2, 3, 8])
    # Check unitary constraint a^2 + b^2 = 1
    unity = a ** 2 + b ** 2
    assert torch.allclose(unity, torch.ones_like(unity), atol=1e-5)


def test_quantum_interference_layer():
    inter = QuantumInterferenceLayer(d_model=8)
    c = torch.randn(2, 8)
    v = torch.randn(2, 8)
    out = inter(c, v)
    assert out.shape == torch.Size([2, 8])


def test_stochastic_attention_and_entropy():
    attn = StochasticQuantumAttention(d_model=8, n_heads=1)
    x = torch.randn(2, 4, 8)
    out, entropy = attn(x)
    assert out.shape == torch.Size([2, 4, 8])
    assert entropy.shape == torch.Size([2, 4])
    assert (entropy >= 0.0).all()


def test_quantum_collapse_engine():
    collapse = QuantumCollapseEngine(d_model=8, layer_idx=1, total_layers=2)
    combined = torch.randn(2, 4, 8)
    c_prime = torch.randn(2, 4, 8)
    entropy = torch.ones(2, 4) * 0.5
    y, tau = collapse(combined, c_prime, entropy)
    assert y.shape == torch.Size([2, 4, 8])
    assert (tau >= 0.0).all() and (tau <= 1.0).all()


def test_quantum_uncertainty_memory():
    mem = QuantumUncertaintyMemory(d_model=8)
    v = torch.randn(2, 8)
    m_prev = torch.zeros(2, 8)
    v_out, m_next = mem(v, m_prev)
    assert v_out.shape == torch.Size([2, 8])
    assert m_next.shape == torch.Size([2, 8])


def test_riemannian_geodesic_engine():
    geo = RiemannianGeodesicEngine(d_model=8, rank_k=4)
    x1 = torch.randn(2, 3, 8)
    x2 = x1 + torch.randn(2, 3, 8) * 0.1
    h = torch.ones(2, 3) * 0.5
    dist = geo.compute_geodesic_distance([x1, x2], [h])
    loss = geo.geodesic_loss(dist)
    assert dist.shape == torch.Size([2, 3])
    assert (dist >= 0.0).all()
    assert loss.item() >= 0.0


def test_quantum_pool_full_cycle():
    pool = QuantumPool(d_model=8, total_layers=2)
    c = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    res = pool.execute_full_cycle(c, v)
    assert "token_superposition" in res
    assert "geodesic_loss" in res
    total_loss = res["collapsed"].sum() + res["geodesic_loss"] + res["token_superposition"].sum()
    total_loss.backward()
    for name, p in pool.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
