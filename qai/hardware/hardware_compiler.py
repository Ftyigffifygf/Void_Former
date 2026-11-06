"""Hardware compilation and transpilation utilities for Qiskit circuits."""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from qiskit import QuantumCircuit, transpile

logger = logging.getLogger(__name__)


class HardwareCompiler:
    """Handles transpilation and physical resource analysis for hardware QPUs."""

    @staticmethod
    def transpile_circuit(
        circuit: QuantumCircuit,
        backend: Any,
        optimization_level: int = 1,
    ) -> QuantumCircuit:
        """Transpiles a Qiskit QuantumCircuit for a target hardware backend."""
        try:
            return transpile(circuit, backend=backend, optimization_level=optimization_level)
        except Exception as e:
            logger.error(f"Transpilation failed: {e}")
            raise

    @staticmethod
    def analyze_transpiled_circuit(circuit: QuantumCircuit) -> Dict[str, Any]:
        """Calculates physical circuit depth, physical gate counts, and physical qubit count."""
        ops = circuit.count_ops()
        return {
            "physical_depth": circuit.depth(),
            "physical_qubits_used": circuit.num_qubits,
            "physical_gate_counts": dict(ops),
            "two_qubit_gates": sum(count for gate, count in ops.items() if gate in ["cx", "cz", "ecr"]),
        }
