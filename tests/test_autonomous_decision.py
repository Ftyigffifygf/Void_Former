"""Unit tests for Quantum Autonomous Decision Simulation Engine & Customizable Argon Harness."""

from __future__ import annotations

import torch

from voidformer.quantum.autonomous_decision import (
    QuantumAutonomousDecisionEngine,
    CustomizableQuantumHarness,
)
from voidformer.quantum.plugin_bridge import QuantumVoidFormerAIPlugin


def test_quantum_autonomous_decision_engine_standalone():
    B, T, D, Q = 2, 8, 64, 4
    engine = QuantumAutonomousDecisionEngine(
        d_model=D, n_vqc_qubits=Q, num_simulations=4, amplification_iterations=2
    )

    x = torch.randn(B, T, D, requires_grad=True)
    out, diag = engine(x)

    assert out.shape == (B, T, D)
    assert diag["num_simulated_trajectories"] == 2 ** Q
    assert len(diag["trajectory_quality_scores"]) == 2

    loss = out.sum()
    loss.backward()

    assert x.grad is not None


def test_quantum_autonomous_decision_engine_custom_evaluator():
    B, T, D, Q = 2, 8, 64, 4
    engine = QuantumAutonomousDecisionEngine(
        d_model=D, n_vqc_qubits=Q, num_simulations=4, amplification_iterations=2
    )

    # Custom domain utility evaluator (e.g. business idea / coding solution feasibility)
    def custom_business_evaluator(probs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(probs.mean(dim=-1, keepdim=True))

    x = torch.randn(B, T, D)
    out, diag = engine(x, custom_evaluator_fn=custom_business_evaluator)

    assert out.shape == (B, T, D)
    assert "num_simulated_trajectories" in diag


def test_customizable_quantum_harness():
    harness = CustomizableQuantumHarness(
        d_model=64, n_vqc_qubits=4, domain_name="business_and_coding_genius"
    )

    x = torch.randn(2, 8, 64)
    out, diag = harness.simulate_task(x)

    assert out.shape == (2, 8, 64)
    assert diag["domain_name"] == "business_and_coding_genius"


def test_quantum_ai_plugin_autonomous_decision():
    plugin = QuantumVoidFormerAIPlugin(
        d_model=64, n_vqc_qubits=4, thinking_steps=2
    )

    x = torch.randn(2, 8, 64)
    out, diag = plugin.simulate_autonomous_decision(x)

    assert out.shape == (2, 8, 64)
    assert "num_simulated_trajectories" in diag
