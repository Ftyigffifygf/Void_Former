"""Unit tests for DeepSeek Quantum Reasoning & Evaluation Harness."""

from __future__ import annotations

import torch
import pytest

from voidformer.harness.deepseek_quantum_harness import (
    DeepSeekQuantumHarness,
    QuantumProcessRewardModel,
    GRPOQuantumRewardNormalizer,
)
from voidformer.harness.cli import run_deepseek_quantum_eval
from voidformer.models.quantum_voidformer import QuantumVoidFormer


def test_quantum_process_reward_model():
    B, T, Q = 2, 8, 4
    prm = QuantumProcessRewardModel(d_model=64, n_vqc_qubits=Q)

    psi = torch.randn(B, T, 2 ** Q, dtype=torch.complex64)
    scores = prm.evaluate_step(psi)

    assert scores.shape == (B, T, 1)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_grpo_quantum_reward_normalizer():
    group_rewards = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    advantages = GRPOQuantumRewardNormalizer.normalize_group_rewards(group_rewards)

    assert advantages.shape == (4, 2)
    assert torch.allclose(advantages.mean(dim=0), torch.zeros(2), atol=1e-5)


def test_deepseek_quantum_harness_evaluation():
    model = QuantumVoidFormer(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        n_qubits_per_token=4,
        n_vqc_qubits=4,
        max_seq_len=64,
    )

    harness = DeepSeekQuantumHarness(d_model=64, n_vqc_qubits=4, group_size=4, thinking_steps=2)

    ids = torch.randint(0, 100, (2, 8))
    logits, diag = harness.evaluate_reasoning_task(model, ids)

    assert logits.shape == (2, 8, 100)
    assert "DeepSeek_Quantum" in diag["harness_name"]
    assert diag["group_size"] == 4
    assert len(diag["group_rewards"]) == 4


def test_harness_cli_deepseek_eval():
    diag = run_deepseek_quantum_eval(steps=2)
    assert "DeepSeek_Quantum" in diag["harness_name"]
    assert "mean_group_reward" in diag
