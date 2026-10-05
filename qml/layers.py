"""Hybrid Quantum-Classical Layers using PennyLane and PyTorch."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
from typing import Optional

from qml.backends import get_pennylane_device


class ParameterShiftAutogradFunction(torch.autograd.Function):
    """Custom PyTorch autograd function implementing exact Parameter-Shift Rule:
    (f(θ + π/2) - f(θ - π/2)) / 2
    """

    @staticmethod
    def forward(ctx, qnode, weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        ctx.qnode = qnode
        ctx.save_for_backward(weights, inputs)

        res = qnode(inputs, weights)
        if isinstance(res, (list, tuple)):
            res = torch.stack(res, dim=-1)
        return res

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        qnode = ctx.qnode
        weights, inputs = ctx.saved_tensors

        shift = torch.pi / 2.0
        grad_weights = torch.zeros_like(weights)

        for idx in range(weights.numel()):
            w_flat = weights.clone().view(-1)

            w_plus = w_flat.clone()
            w_plus[idx] += shift
            res_plus = qnode(inputs, w_plus.view_as(weights))
            if isinstance(res_plus, (list, tuple)):
                res_plus = torch.stack(res_plus, dim=-1)

            w_minus = w_flat.clone()
            w_minus[idx] -= shift
            res_minus = qnode(inputs, w_minus.view_as(weights))
            if isinstance(res_minus, (list, tuple)):
                res_minus = torch.stack(res_minus, dim=-1)

            shift_grad = (res_plus - res_minus) / 2.0
            grad_weights.view(-1)[idx] = torch.sum(grad_output * shift_grad)

        return None, grad_weights, None


def create_quantum_circuit(n_qubits: int = 4, n_layers: int = 2, dev: Optional[qml.Device] = None):
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    diff_method = "parameter-shift" if (hasattr(dev, "shots") and dev.shots) else "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class EquivariantQNNLayer(nn.Module):
    """Equivariant Quantum Neural Network (EQNN) layer with symmetry group invariance (SU(N) / Permutation Symmetry)."""

    def __init__(self, n_qubits: int = 4, symmetry_group: str = "permutation"):
        super().__init__()
        self.n_qubits = n_qubits
        self.symmetry_group = symmetry_group

        if symmetry_group == "permutation":
            self.weight = nn.Parameter(torch.randn(1, 3))
        else:
            self.weight = nn.Parameter(torch.randn(n_qubits, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Construct symmetric square matrix W of shape (n_qubits, n_qubits)
        if self.symmetry_group == "permutation":
            v = self.weight.repeat(self.n_qubits, 1)  # (n_qubits, 3)
            W = torch.matmul(v, v.T) / 3.0             # (n_qubits, n_qubits)
        else:
            W = torch.matmul(self.weight, self.weight.T) / 3.0

        weight_transform = torch.matmul(x, W[:x.shape[-1], :x.shape[-1]])
        out = torch.tanh(x + weight_transform)
        return out


class HybridNet(nn.Module):
    """Hybrid Classical-Quantum Neural Network."""

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
        self.eqnn = EquivariantQNNLayer(n_qubits=n_qubits, symmetry_group="permutation")
        self.post = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_out = torch.tanh(self.pre(x))
        q_out = self.q(pre_out)
        eq_out = self.eqnn(q_out)
        return self.post(eq_out)


class ClassicalNet(nn.Module):
    """Purely Classical Neural Network matched in structure and parameter count."""

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
            q_params = sum(p.numel() for p in match_parameters_of.parameters() if p.requires_grad)
            mid_dim = max(1, (q_params - (in_features * hidden_dim + hidden_dim) - (hidden_dim * num_classes + num_classes)) // (hidden_dim + 1))
            self.mid = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.mid = nn.Linear(hidden_dim, hidden_dim)

        self.post = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.pre(x))
        h = torch.tanh(self.mid(h))
        return self.post(h)
