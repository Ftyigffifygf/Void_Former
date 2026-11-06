"""Training module for Q-INTELLIGENCE."""

from qai.training.loss import get_loss_function
from qai.training.optimizer import create_optimizer
from qai.training.metrics import compute_metrics
from qai.training.trainer import Trainer
from qai.training.quantum_trainer import QuantumTrainer
from qai.training.classical_trainer import Trainer as ClassicalTrainer

__all__ = [
    "get_loss_function",
    "create_optimizer",
    "compute_metrics",
    "Trainer",
    "QuantumTrainer",
    "ClassicalTrainer",
]
