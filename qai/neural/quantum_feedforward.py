"""Quantum Feed-Forward Block with classical projections and data re-uploading."""

from __future__ import annotations
import torch
import torch.nn as nn
import pennylane as qml
from typing import Optional
from qai.core.device_manager import DeviceManager
from qai.core.tensor_utils import normalize_quantum_features
from qai.quantum.encoding import QuantumEncoder
from qai.quantum.ansatz import VariationalAnsatz
from qai.quantum.measurements import QuantumMeasurements


class QuantumFeedForward(nn.Module):
    """
    Quantum Feed-Forward Network:
    Transformer Representation -> Classical Down Projection -> Quantum Encoding (with optional re-uploading)
    -> Trainable Variational Circuit -> Measurement -> Classical Up Projection -> Residual Connection
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 16,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend_name: str = "default.qubit",
        shots: Optional[int] = None,
        use_reuploading: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_reuploading = use_reuploading

        self.down_proj = nn.Linear(d_model, n_qubits)
        self.up_proj = nn.Linear(n_qubits, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Variational parameters
        # If re-uploading: (n_layers, n_qubits, 3)
        rot_dim = 3 if use_reuploading else 2
        self.vqc_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, rot_dim) * 0.05
        )

        self.device = DeviceManager.create_device(
            backend_name=backend_name,
            n_qubits=n_qubits,
            shots=shots,
        )

        wires = list(range(n_qubits))

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def _circuit(x_vec: torch.Tensor, weights: torch.Tensor):
            if use_reuploading:
                for l in range(n_layers):
                    QuantumEncoder.data_reuploading_step(x_vec, weights[l], wires)
                    # Entangling layer
                    for q in range(n_qubits):
                        qml.CNOT(wires=[wires[q], wires[(q + 1) % n_qubits]])
            else:
                QuantumEncoder.angle_embedding(x_vec, wires)
                VariationalAnsatz.hardware_efficient(weights, wires, entanglement="ring")

            return QuantumMeasurements.pauli_z_expectations(wires)

        self._circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual input
        residual = x

        # Down projection & normalization
        down = self.down_proj(x)
        norm_down = normalize_quantum_features(down)

        orig_shape = norm_down.shape
        flat = norm_down.view(-1, self.n_qubits)

        q_outputs = []
        for i in range(flat.shape[0]):
            meas = self._circuit(flat[i], self.vqc_weights)
            meas_t = torch.stack(meas) if isinstance(meas, (list, tuple)) else meas
            q_outputs.append(meas_t)

        q_out = torch.stack(q_outputs).view(*orig_shape[:-1], self.n_qubits).to(dtype=self.up_proj.weight.dtype)
        up = self.up_proj(q_out)
        return self.norm(residual + up)
