"""Hybrid Quantum-Classical Layers using PennyLane and PyTorch."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
from typing import Optional

from qml.backends import get_pennylane_device


def create_quantum_circuit(n_qubits: int = 4, n_layers: int = 2, dev: Optional[qml.Device] = None):
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    # Use backprop diff_method for exact statevector devices (simulator)
    # and parameter-shift for hardware / finite shots devices
    diff_method = "parameter-shift" if (hasattr(dev, "shots") and dev.shots) else "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class HybridNet(nn.Module):
    """Hybrid Classical-Quantum Neural Network.

    Architecture:
      Classical Linear (in_features -> n_qubits) + Tanh activation
      Quantum Circuit Layer (n_qubits, trainable StronglyEntanglingLayers) -> Pauli-Z expvals
      Classical Linear (n_qubits -> num_classes)
    """

    def __init__(
        self,
        in_features: int = 16,
        n_qubits: int = 4,
        n_layers: int = 2,
        num_classes: int = 2,
        backend_kind: str = "simulator",
        dev: Optional[qml.Device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if dev is None:
            dev = get_pennylane_device(backend_kind, n_qubits=n_qubits)

        self.qnode = create_quantum_circuit(n_qubits=n_qubits, n_layers=n_layers, dev=dev)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        self.pre = nn.Linear(in_features, n_qubits)
        self.q = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.post = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_out = torch.tanh(self.pre(x))
        q_out = self.q(pre_out)
        return self.post(q_out)


class ClassicalNet(nn.Module):
    """Purely Classical Neural Network matched in structure and parameter count.

    Architecture:
      Linear (in_features -> hidden_dim) + Tanh
      Linear (hidden_dim -> hidden_dim) + Tanh
      Linear (hidden_dim -> num_classes)
    """

    def __init__(
        self,
        in_features: int = 16,
        hidden_dim: int = 4,
        num_classes: int = 2,
        match_parameters_of: Optional[HybridNet] = None,
    ):
        super().__init__()
        self.pre = nn.Linear(in_features, hidden_dim)

        if match_parameters_of is not None:
            # Count trainable parameters in match_parameters_of
            q_params = sum(p.numel() for p in match_parameters_of.parameters() if p.requires_grad)
            # Match parameter count with a hidden layer
            mid_dim = max(1, (q_params - (in_features * hidden_dim + hidden_dim) - (hidden_dim * num_classes + num_classes)) // (hidden_dim + 1))
            self.mid = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.mid = nn.Linear(hidden_dim, hidden_dim)

        self.post = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.pre(x))
        h = torch.tanh(self.mid(h))
        return self.post(h)
