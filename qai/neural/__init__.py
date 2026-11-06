"""Neural network modules for Q-INTELLIGENCE."""

from qai.neural.classical_ann import ClassicalANN
from qai.neural.quantum_ann import QuantumANN
from qai.neural.hybrid_ann import HybridANN
from qai.neural.quantum_feedforward import QuantumFeedForward

__all__ = [
    "ClassicalANN",
    "QuantumANN",
    "HybridANN",
    "QuantumFeedForward",
]
