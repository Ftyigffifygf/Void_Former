"""Circuit builder and resource estimator for variational circuits."""

from __future__ import annotations
import torch
import pennylane as qml
from typing import Sequence, Dict, Any, Optional
from qai.quantum.encoding import QuantumEncoder
from qai.quantum.ansatz import VariationalAnsatz
from qai.quantum.measurements import QuantumMeasurements


class CircuitBuilder:
    """Constructs PennyLane QNodes with resource tracking."""

    def __init__(
        self,
        device: qml.devices.Device,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "angle",
        entanglement: str = "ring",
        diff_method: str = "backprop",
    ) -> None:
        self.device = device
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.entanglement = entanglement
        self.diff_method = diff_method
        self.wires = list(range(n_qubits))

    def build_qnode(self, measurement_type: str = "expval_z") -> Any:
        wires = self.wires
        encoding_type = self.encoding
        entanglement = self.entanglement

        @qml.qnode(self.device, interface="torch", diff_method=self.diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # Encode inputs
            if encoding_type == "iqp":
                QuantumEncoder.iqp_embedding(inputs, wires, pattern=entanglement)
            elif encoding_type == "amplitude":
                QuantumEncoder.amplitude_embedding(inputs, wires)
            else:
                QuantumEncoder.angle_embedding(inputs, wires, rotation="Y")

            # Variational ansatz
            VariationalAnsatz.hardware_efficient(weights, wires, entanglement=entanglement)

            # Measurements
            if measurement_type == "probs":
                return QuantumMeasurements.all_probabilities(wires)
            else:
                return QuantumMeasurements.pauli_z_expectations(wires)

        return circuit

    def estimate_resources(self) -> Dict[str, Any]:
        """Provides logical resource estimates for the configured circuit."""
        # Calculate gate counts logically
        # 1. Encoding:
        encoding_gates = self.n_qubits if self.encoding == "angle" else (self.n_qubits * 2 + (self.n_qubits if self.entanglement == "ring" else self.n_qubits - 1))
        # 2. Ansatz:
        rotations_per_layer = self.n_qubits * 2  # RY + RZ
        entangling_per_layer = self.n_qubits if self.entanglement == "ring" else (self.n_qubits - 1)
        total_ansatz_rotations = self.n_layers * rotations_per_layer
        total_ansatz_entangling = self.n_layers * entangling_per_layer
        total_gates = encoding_gates + total_ansatz_rotations + total_ansatz_entangling

        # Trainable parameters
        trainable_params = self.n_layers * self.n_qubits * 2

        # Depth estimate
        depth_per_layer = 2 + (2 if self.entanglement == "ring" else 1)
        circuit_depth = 1 + self.n_layers * depth_per_layer

        return {
            "n_logical_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "entanglement": self.entanglement,
            "trainable_quantum_parameters": trainable_params,
            "single_qubit_gates": encoding_gates + total_ansatz_rotations,
            "entangling_gates": total_ansatz_entangling,
            "total_logical_gates": total_gates,
            "logical_circuit_depth": circuit_depth,
            "backend_name": getattr(self.device, "name", str(self.device)),
            "shots": getattr(self.device, "shots", None),
        }
