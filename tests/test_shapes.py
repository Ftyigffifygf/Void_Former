"""Shape & forward-pass sanity tests for VoidFormer layers and the full model."""

from __future__ import annotations

import torch

from voidformer.layers import (
    AdaptiveCollapseEngine,
    ClassicalAttention,
    DualEmbedding,
    DynamicRouter,
    InterferenceAttention,
    InterferenceEngine,
    UncertaintyMemory,
    VoidAttention,
    VoidFormerBlock,
)
from voidformer.models import VoidFormerModel


B, T, V = 2, 16, 64
DM, DV, H, FF = 32, 32, 4, 64


def test_dual_embedding_shapes():
    emb = DualEmbedding(V, DM, DV, max_seq_len=64)
    ids = torch.randint(0, V, (B, T))
    c, v = emb(ids)
    assert c.shape == (B, T, DM)
    assert v.shape == (B, T, DV)


def test_classical_attention_shape():
    attn = ClassicalAttention(DM, H)
    x = torch.randn(B, T, DM)
    y, w = attn(x)
    assert y.shape == x.shape
    assert w.shape == (B, H, T, T)


def test_void_attention_returns_entropy():
    attn = VoidAttention(DV, H, noise_std=0.0)
    x = torch.randn(B, T, DV)
    y, w, ent = attn(x)
    assert y.shape == x.shape
    assert ent.shape == (B, H, T)
    assert torch.isfinite(ent).all()
    assert (ent >= 0).all()


def test_interference_attention_shape():
    ia = InterferenceAttention(DM, DV, H)
    c = torch.randn(B, T, DM)
    v = torch.randn(B, T, DV)
    ip, in_ = ia(c, v)
    assert ip.shape == (B, T, DM)
    assert in_.shape == (B, T, DM)


def test_interference_engine_shape():
    eng = InterferenceEngine(DM, DV)
    c = torch.randn(B, T, DM); v = torch.randn(B, T, DV)
    ip = torch.randn(B, T, DM); in_ = torch.randn(B, T, DM)
    out, coefs = eng(c, v, ip, in_)
    assert out.shape == (B, T, DM)
    assert coefs.shape == (B, T, 5)


def test_collapse_engine_tau_in_range():
    ce = AdaptiveCollapseEngine(DM)
    c = torch.randn(B, T, DM); comb = torch.randn(B, T, DM)
    ent = torch.rand(B, T)
    y, tau = ce(c, comb, ent, depth_ratio=0.5, mode="soft")
    assert y.shape == (B, T, DM)
    assert tau.shape == (B, T)
    assert (tau >= 0).all() and (tau <= 1).all()


def test_uncertainty_memory_shapes():
    um = UncertaintyMemory(DV)
    v = torch.randn(B, T, DV)
    m0 = um.init_state(B, T, DV, v.device)
    out, m1 = um(v, m0)
    assert out.shape == v.shape == m1.shape


def test_dynamic_router_weights_sum_to_one():
    dr = DynamicRouter(DM)
    dr.eval()
    c = torch.randn(B, T, DM); v = torch.randn(B, T, DM); m = torch.randn(B, T, DM)
    out, w = dr(c, v, m)
    assert out.shape == (B, T, DM)
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5)


def test_voidformer_block_forward():
    block = VoidFormerBlock(
        d_model=DM, d_void=DV, n_heads=H, d_ff=FF, dropout=0.0,
        void_noise_std=0.0, collapse_temperature_init=1.0,
        use_uncertainty_memory=True, use_dynamic_router=True,
    )
    c = torch.randn(B, T, DM); v = torch.randn(B, T, DV)
    mem = torch.zeros(B, T, DV)
    c2, v2, mem2, diag = block(c, v, mem, depth_ratio=0.5, mode="adaptive-collapse")
    assert c2.shape == c.shape
    assert v2.shape == v.shape
    assert mem2.shape == mem.shape
    assert "tau" in diag and "void_entropy" in diag and "coefs" in diag


def test_voidformer_model_forward_and_diagnostics():
    model = VoidFormerModel(
        vocab_size=V, d_model=DM, d_void=DV, n_layers=2,
        n_heads=H, d_ff=FF, max_seq_len=T, dropout=0.0,
        void_noise_std=0.0,
    )
    ids = torch.randint(0, V, (B, T))
    out = model(ids, return_diagnostics=True)
    assert out.logits.shape == (B, T, V)
    assert out.classical_states.shape == (B, T, DM)
    assert out.void_states.shape == (B, T, DV)
    assert len(out.diagnostics) == 2


def test_voidformer_model_modes():
    model = VoidFormerModel(
        vocab_size=V, d_model=DM, d_void=DV, n_layers=2,
        n_heads=H, d_ff=FF, max_seq_len=T, dropout=0.0, void_noise_std=0.0,
    )
    ids = torch.randint(0, V, (B, T))
    for mode in [
        "deterministic", "probabilistic",
        "interference-heavy", "adaptive-collapse", "latent-memory",
    ]:
        model.set_mode(mode)
        out = model(ids)
        assert torch.isfinite(out.logits).all(), f"non-finite logits in mode={mode}"
