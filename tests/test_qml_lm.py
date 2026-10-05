"""Unit tests for small Classical LM vs Hybrid Quantum LM."""

import pytest
import torch
from torch.utils.data import DataLoader
from qml.lm import (
    ClassicalTransformerLM,
    HybridQuantumTransformerLM,
    QuantumFFNBlock,
    compute_perplexity,
)
from qml.train_lm import CharDataset, train_lm, SAMPLE_TEXT


def test_quantum_ffn_block_forward():
    block = QuantumFFNBlock(d_model=32, n_qubits=4, n_layers=1)
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.shape == (2, 8, 32)


def test_classical_and_hybrid_lm_forward():
    c_lm = ClassicalTransformerLM(vocab_size=64, d_model=32, n_heads=2, n_layers=1, d_ff=64)
    q_lm = HybridQuantumTransformerLM(vocab_size=64, d_model=32, n_heads=2, n_layers=1, n_qubits=4)

    ids = torch.randint(0, 64, (2, 16))
    out_c = c_lm(ids)
    out_q = q_lm(ids)

    assert out_c.shape == (2, 16, 64)
    assert out_q.shape == (2, 16, 64)


def test_char_dataset_and_training_loop():
    dataset = CharDataset(text=SAMPLE_TEXT[:200], seq_len=16)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    q_lm = HybridQuantumTransformerLM(vocab_size=dataset.vocab_size, d_model=32, n_heads=2, n_layers=1)
    val_loss, val_ppl, train_loss = train_lm(q_lm, train_loader, val_loader, epochs=1, lr=0.01)

    assert isinstance(val_loss, float)
    assert isinstance(val_ppl, float)
    assert val_ppl >= 1.0


def test_compute_perplexity_func():
    assert compute_perplexity(0.0) == pytest.approx(1.0)
    assert compute_perplexity(1.0) == pytest.approx(2.71828, abs=1e-3)
