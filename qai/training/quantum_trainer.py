"""Quantum and hardware-in-the-loop training utilities with shot safeguards."""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import torch
from qai.training.trainer import Trainer

logger = logging.getLogger(__name__)


class QuantumTrainer(Trainer):
    """Specialized trainer for quantum and hardware execution with execution budgeting."""

    def __init__(self, *args, max_circuit_executions: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_circuit_executions = max_circuit_executions
        self.circuit_execution_count = 0

    def estimate_shots_required(self, shots_per_eval: int = 1024) -> int:
        """Estimates total shots across the entire training run."""
        batches_per_epoch = len(self.train_loader)
        total_evals = batches_per_epoch * self.epochs
        return total_evals * shots_per_eval
