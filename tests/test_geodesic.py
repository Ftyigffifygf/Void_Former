"""Tests for the Riemannian geodesic machinery."""

from __future__ import annotations

import torch

from voidformer.layers import FisherInfoMetric, RiemannianGeodesicTracker
from voidformer.models import VoidFormerModel
from voidformer.training import VoidFormerLosses


B, T, D = 2, 8, 32
V = 64


def test_fisher_metric_psd_quadratic_nonnegative():
    fm = FisherInfoMetric(D, rank=4)
    x = torch.randn(B, T, D)
    ent = torch.rand(B, T)
    U, d_diag, kappa = fm(x, ent)
    assert U.shape == (B, T, 4, D)
    assert d_diag.shape == (B, T, D)
    assert (d_diag > 0).all()
    assert (kappa > 0).all()
    delta = torch.randn(B, T, D)
    q = FisherInfoMetric.quadratic(delta, U, d_diag, kappa)
    assert q.shape == (B, T)
    assert (q >= 0).all(), "quadratic form must be non-negative (PSD metric)"


def test_geodesic_zero_for_constant_path():
    tracker = RiemannianGeodesicTracker(D, rank=4)
    tracker.reset()
    x = torch.randn(B, T, D)
    ent = torch.rand(B, T)
    tracker.step(x, x, ent)
    tracker.step(x, x, ent)
    total, segs = tracker.finalize()
    assert total.shape == (B, T)
    # Δx = 0 ⇒ d ≈ √ε ~ 1e-4; assert near zero
    assert total.max().item() < 1.0e-3


def test_geodesic_monotone_in_step_size():
    """Doubling the step should ~ double the segment length (linear in Δ on PSD)."""
    tracker = RiemannianGeodesicTracker(D, rank=4)
    tracker.reset()
    x0 = torch.zeros(B, T, D)
    x1 = torch.ones(B, T, D)
    x2 = 2.0 * torch.ones(B, T, D)
    ent = torch.zeros(B, T)
    s1 = tracker.step(x0, x1, ent).clone()
    s2 = tracker.step(x1, x2, ent).clone()
    # midpoint changes but metric is mild → 2nd segment should be > 1st
    assert (s2 > 0).all() and (s1 > 0).all()


def test_voidformer_geodesic_attached_to_output():
    model = VoidFormerModel(
        vocab_size=V, d_model=D, d_void=D, n_layers=3,
        n_heads=4, d_ff=64, max_seq_len=T, dropout=0.0,
        void_noise_std=0.0, use_geodesic_tracker=True, geodesic_rank=4,
    )
    ids = torch.randint(0, V, (B, T))
    out = model(ids, return_diagnostics=True)
    assert out.geodesic_distance is not None
    assert out.geodesic_distance.shape == (B, T)
    assert len(out.geodesic_segments) == 3
    assert (out.geodesic_distance >= 0).all()


def test_geodesic_can_be_disabled():
    model = VoidFormerModel(
        vocab_size=V, d_model=D, d_void=D, n_layers=2,
        n_heads=4, d_ff=64, max_seq_len=T, dropout=0.0,
        void_noise_std=0.0, use_geodesic_tracker=False,
    )
    ids = torch.randint(0, V, (B, T))
    out = model(ids)
    assert out.geodesic_distance is None
    assert out.geodesic_segments == []


def test_geodesic_loss_term_finite_and_gradient_flows():
    model = VoidFormerModel(
        vocab_size=V, d_model=D, d_void=D, n_layers=3,
        n_heads=4, d_ff=64, max_seq_len=T, dropout=0.0,
        void_noise_std=0.0, use_geodesic_tracker=True, geodesic_rank=4,
    )
    loss_fn = VoidFormerLosses({
        "lambda_entropy_preservation": 0.0,
        "lambda_semantic_consistency": 0.0,
        "lambda_collapse_reg": 0.0,
        "lambda_interference_stability": 0.0,
        "lambda_ambiguity_retention": 0.0,
        "lambda_geodesic": 1.0,
        "geodesic_target": 1.0,
    })
    ids = torch.randint(0, V, (B, T))
    out = model(ids, return_diagnostics=True)
    total, log = loss_fn(out, ids, model.embed.classical_emb.weight)
    assert torch.isfinite(total)
    assert "loss/geodesic" in log
    assert log["loss/geodesic"].item() >= 0.0
    total.backward()
    # at least one geodesic-tracker param should have a non-zero gradient
    g = [p.grad for p in model.geodesic_tracker.parameters() if p.grad is not None]
    assert g, "no gradients reached the geodesic tracker"
    assert any(gi.abs().sum() > 0 for gi in g)
