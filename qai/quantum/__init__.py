"""Quantum computing engine module."""

from qai.quantum.encoding import QuantumEncoder
from qai.quantum.entanglement import EntanglementBuilder
from qai.quantum.ansatz import VariationalAnsatz
from qai.quantum.measurements import QuantumMeasurements
from qai.quantum.circuit_builder import CircuitBuilder
from qai.quantum.gradients import compute_gradient_norms

__all__ = [
    "QuantumEncoder",
    "EntanglementBuilder",
    "VariationalAnsatz",
    "QuantumMeasurements",
    "CircuitBuilder",
    "compute_gradient_norms",
    "QuantumPool",
    "QuantumTokenSuperposition",
    "QuantumInterferenceLayer",
    "StochasticQuantumAttention",
    "QuantumCollapseEngine",
    "QuantumUncertaintyMemory",
    "RiemannianGeodesicEngine",
]

from qai.quantum.quantum_pool import (
    QuantumPool,
    QuantumTokenSuperposition,
    QuantumInterferenceLayer,
    StochasticQuantumAttention,
    QuantumCollapseEngine,
    QuantumUncertaintyMemory,
    RiemannianGeodesicEngine,
)
