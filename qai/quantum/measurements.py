"""Quantum measurement and observable extraction."""

from __future__ import annotations
import pennylane as qml
from typing import Sequence, List, Any


class QuantumMeasurements:
    """Provides methods for extracting expectation values and probabilities."""

    @staticmethod
    def pauli_z_expectations(wires: Sequence[int]) -> List[Any]:
        """Returns expectation values of Pauli-Z on each wire."""
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    @staticmethod
    def pauli_x_expectations(wires: Sequence[int]) -> List[Any]:
        """Returns expectation values of Pauli-X on each wire."""
        return [qml.expval(qml.PauliX(w)) for w in wires]

    @staticmethod
    def pauli_y_expectations(wires: Sequence[int]) -> List[Any]:
        """Returns expectation values of Pauli-Y on each wire."""
        return [qml.expval(qml.PauliY(w)) for w in wires]

    @staticmethod
    def all_probabilities(wires: Sequence[int]) -> Any:
        """Returns the full computational basis state probabilities (shape: 2^n)."""
        return qml.probs(wires=wires)

    @staticmethod
    def local_probabilities(wires: Sequence[int]) -> List[Any]:
        """Returns single-qubit marginal probabilities for each wire."""
        return [qml.probs(wires=w) for w in wires]
