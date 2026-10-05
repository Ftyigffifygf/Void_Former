"""DeepSeek Harness Model Plugin Interface."""

from __future__ import annotations

from typing import Dict, Any
import torch
import torch.nn as nn

from voidformer.harness.deepseek_quantum_harness import DeepSeekQuantumHarness


class DeepSeekHarnessModelPlugin:
    """Model plugin interface connecting DeepSeek Quantum Harness with serve API endpoints."""

    def __init__(self, d_model: int = 256, n_vqc_qubits: int = 8, group_size: int = 4):
        self.harness = DeepSeekQuantumHarness(
            d_model=d_model,
            n_vqc_qubits=n_vqc_qubits,
            group_size=group_size,
        )

    def evaluate_model_plugin(self, model: nn.Module, input_ids: torch.Tensor) -> Dict[str, Any]:
        logits, diag = self.harness.evaluate_reasoning_task(model, input_ids)
        return diag
