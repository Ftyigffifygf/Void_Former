"""Quantum Artificial Neural Network (QANN) architecture."""

from __future__ import annotations
import torch
import torch.nn as nn
import pennylane as qml
from typing import Sequence, Optional
from qai.core.device_manager import DeviceManager
from qai.core.tensor_utils import normalize_quantum_features
from qai.quantum.encoding import QuantumEncoder
from qai.quantum.ansatz import VariationalAnsatz
from qai.quantum.measurements import QuantumMeasurements


class QuantumANN(nn.Module):
    """
    Quantum Artificial Neural Network:
    Input -> Normalization -> Classical Linear Projection -> Quantum Data Encoding
    -> Parameterized Quantum Circuit -> Entanglement Layers -> Quantum Measurement
    -> Classical Output Projection -> Prediction
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend_name: str = "default.qubit",
        encoding: str = "angle",
        entanglement: str = "ring",
        diff_method: Optional[str] = None,
        task: str = "classification",  # 'classification' or 'regression'
        shots: Optional[int] = None,
        shots_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.task = task
        self.encoding = encoding
        self.entanglement = entanglement

        # Classical pre-projection
        self.input_proj = nn.Linear(in_features, n_qubits)

        # Trainable quantum weights (n_layers, n_qubits, 2)
        # Identity-near initialization
        self.quantum_weights = nn.Parameter(
            VariationalAnsatz.identity_near_init(n_layers, n_qubits, n_rotations=2, scale=0.05)
        )

        # Classical output projection
        self.output_proj = nn.Linear(n_qubits, out_features)

        # Initialize PennyLane device
        self.device = DeviceManager.create_device(
            backend_name=backend_name,
            n_qubits=n_qubits,
            shots=shots or shots_override,
        )

        # Select appropriate diff_method based on device if not specified
        if diff_method is None:
            if "lightning" in backend_name.lower():
                diff_method = "adjoint"
            elif "remote" in backend_name.lower() or shots is not None:
                diff_method = "parameter-shift"
            else:
                diff_method = "backprop"

        # Build QNode
        wires = list(range(n_qubits))

        @qml.qnode(self.device, interface="torch", diff_method=diff_method)
        def _circuit(x_vec: torch.Tensor, q_weights: torch.Tensor):
            if encoding == "iqp":
                QuantumEncoder.iqp_embedding(x_vec, wires, pattern=entanglement)
            elif encoding == "amplitude":
                QuantumEncoder.amplitude_embedding(x_vec, wires)
            else:
                QuantumEncoder.angle_embedding(x_vec, wires, rotation="Y")

            VariationalAnsatz.hardware_efficient(q_weights, wires, entanglement=entanglement)
            return QuantumMeasurements.pauli_z_expectations(wires)

        self._circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Classical feature projection
        x_proj = self.input_proj(x)
        # 2. Normalization
        x_norm = normalize_quantum_features(x_proj)

        # 3. Quantum circuit forward pass (batched or sample-wise)
        batch_size = x_norm.shape[0] if x_norm.dim() > 1 else 1
        flat_x = x_norm.view(-1, self.n_qubits)

        q_outputs = []
        for i in range(flat_x.shape[0]):
            sample = flat_x[i]
            # Execute QNode
            meas = self._circuit(sample, self.quantum_weights)
            # Stack into tensor
            if isinstance(meas, (list, tuple)):
                meas_t = torch.stack(meas)
            else:
                meas_t = meas
            q_outputs.append(meas_t)

        q_out = torch.stack(q_outputs)
        if x_norm.dim() > 1:
            q_out = q_out.view(*x_norm.shape[:-1], self.n_qubits)
        else:
            q_out = q_out.squeeze(0)

        # Ensure float32 for PyTorch linear layer consistency
        q_out = q_out.to(dtype=self.output_proj.weight.dtype)
        # 4. Classical output projection
        out = self.output_proj(q_out)
        if self.task == "regression" and self.out_features == 1:
            return out.squeeze(-1)
        return out
