"""Quantum Attention Mechanism based on quantum state fidelity overlap circuits."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import pennylane as qml
from typing import Optional
from qai.core.device_manager import DeviceManager
from qai.core.tensor_utils import normalize_quantum_features


class QuantumAttention(nn.Module):
    r"""
    Quantum Attention Mechanism:
    Evaluates similarity between query and key states using a parameterized overlap circuit:
      |psi_theta(x)> = U_theta(x)|0>^{\otimes n}
      F(q, k) = |<psi_theta(k)|psi_theta(q)>|^2
    Implemented via:
      1. Encode query state U_theta(q)|0>
      2. Apply inverse key state U_theta(k)^dagger
      3. Measure probability of returning to |0...0>
      4. Convert probability to attention logit
      5. Apply causal mask, softmax, and aggregate classical values.
    Interleaves data-dependent rotations with trainable gates to prevent unitary cancellation.
    """

    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        backend_name: str = "default.qubit",
        shots: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits

        # Classical projections to map d_model to n_qubits
        self.q_proj = nn.Linear(d_model, n_qubits)
        self.k_proj = nn.Linear(d_model, n_qubits)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Trainable parameters interleaved in the state preparation
        # Shape: (n_qubits, 2)
        self.theta = nn.Parameter(torch.randn(n_qubits, 2) * 0.05)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize PennyLane device
        self.device = DeviceManager.create_device(backend_name, n_qubits=n_qubits, shots=shots)

        wires = list(range(n_qubits))
        diff_method = "adjoint" if "lightning" in backend_name.lower() else "backprop"

        @qml.qnode(self.device, interface="torch", diff_method=diff_method)
        def _overlap_circuit(q_feat: torch.Tensor, k_feat: torch.Tensor, weights: torch.Tensor):
            # Forward: Encode Query with interleaved trainable gates
            for i in range(n_qubits):
                qml.RY(q_feat[i], wires=wires[i])
                qml.RZ(weights[i, 0], wires=wires[i])

            # Entangling coupling
            for i in range(n_qubits):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])

            for i in range(n_qubits):
                qml.RY(weights[i, 1] * q_feat[i], wires=wires[i])

            # Inverse: Uncompute Key with inverted trainable gates
            for i in reversed(range(n_qubits)):
                qml.RY(-weights[i, 1] * k_feat[i], wires=wires[i])

            for i in reversed(range(n_qubits)):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])

            for i in reversed(range(n_qubits)):
                qml.RZ(-weights[i, 0], wires=wires[i])
                qml.RY(-k_feat[i], wires=wires[i])

            # Return probability of all-zeros state |0...0>
            return qml.probs(wires=wires)

        self._overlap_circuit = _overlap_circuit

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = normalize_quantum_features(self.q_proj(x))
        k = normalize_quantum_features(self.k_proj(x))
        v = self.v_proj(x)

        # Calculate pairwise quantum fidelities
        # scores matrix: [batch_size, seq_len, seq_len]
        fidelity_scores = torch.zeros(batch_size, seq_len, seq_len, dtype=x.dtype, device=x.device)

        for b in range(batch_size):
            for i in range(seq_len):
                # Apply causal restriction during calculation to minimize circuit calls
                for j in range(i + 1):
                    probs = self._overlap_circuit(q[b, i], k[b, j], self.theta)
                    p0 = probs[0] if isinstance(probs, torch.Tensor) else probs[0]
                    # Convert probability [0, 1] to logit score: log(p0 + eps)
                    score = torch.log(torch.clamp(p0, min=1e-6, max=1.0))
                    fidelity_scores[b, i, j] = score.to(x.dtype)

        # Set future tokens to -inf
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        fidelity_scores = fidelity_scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(fidelity_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        return self.out_proj(context)
