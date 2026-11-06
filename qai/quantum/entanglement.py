"""Entanglement topologies and layers."""

from __future__ import annotations
import pennylane as qml
from typing import Sequence, Optional


class EntanglementBuilder:
    """Constructs entanglement patterns across qubits."""

    @staticmethod
    def linear_entanglement(wires: Sequence[int], gate: str = "CNOT") -> None:
        """Applies entangling gates linearly: (0, 1), (1, 2), ..., (n-2, n-1)."""
        n = len(wires)
        for i in range(n - 1):
            if gate.upper() == "CZ":
                qml.CZ(wires=[wires[i], wires[i + 1]])
            else:
                qml.CNOT(wires=[wires[i], wires[i + 1]])

    @staticmethod
    def ring_entanglement(wires: Sequence[int], gate: str = "CNOT") -> None:
        """Applies entangling gates in a ring: (0, 1), (1, 2), ..., (n-1, 0)."""
        n = len(wires)
        if n < 2:
            return
        for i in range(n):
            target = (i + 1) % n
            if gate.upper() == "CZ":
                qml.CZ(wires=[wires[i], wires[target]])
            else:
                qml.CNOT(wires=[wires[i], wires[target]])

    @staticmethod
    def parameterized_entanglement(wires: Sequence[int], weights: Sequence, pattern: str = "ring") -> None:
        """Applies parameterized CRX or controlled rotation gates."""
        n = len(wires)
        if n < 2:
            return
        for i in range(n):
            if pattern == "linear" and i == n - 1:
                break
            target = (i + 1) % n
            w = weights[i % len(weights)]
            qml.CRX(w, wires=[wires[i], wires[target]])
