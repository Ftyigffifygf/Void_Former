"""Transformer models: Classical (Model A), Hybrid Quantum (Model B), and Quantum-Native (Model C)."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import pennylane as qml
from typing import Optional, Dict, Any

from qai.transformer.embeddings import TokenPositionalEmbedding
from qai.transformer.classical_attention import ClassicalAttention
from qai.transformer.quantum_attention import QuantumAttention
from qai.neural.quantum_feedforward import QuantumFeedForward
from qai.core.device_manager import DeviceManager
from qai.core.tensor_utils import normalize_quantum_features


# ==========================================
# Model A: Classical Transformer Baseline
# ==========================================
class ClassicalTransformer(nn.Module):
    """
    Model A: Classical Transformer
    Token + Positional Embeddings -> Classical MHA -> FFN -> Output Projection
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 16,
        n_heads: int = 1,
        d_ff: int = 32,
        max_seq_len: int = 16,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = TokenPositionalEmbedding(vocab_size, d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.attn = ClassicalAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        # Attention + Residual + LN
        attn_out = self.attn(x, mask=mask)
        x = self.norm1(x + attn_out)
        # FFN + Residual + LN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return self.head(x)


# ==========================================
# Model B: Hybrid Quantum Transformer
# ==========================================
class HybridQuantumTransformer(nn.Module):
    """
    Model B: Hybrid Quantum Transformer
    Token + Positional Embeddings -> Quantum Fidelity Attention -> Quantum Feed-Forward Block -> Output Projection
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 16,
        n_qubits: int = 4,
        d_ff: int = 16,
        max_seq_len: int = 16,
        dropout: float = 0.0,
        backend_name: str = "default.qubit",
        shots: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = TokenPositionalEmbedding(vocab_size, d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.quantum_attn = QuantumAttention(d_model=d_model, n_qubits=n_qubits, backend_name=backend_name, shots=shots, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.quantum_ffn = QuantumFeedForward(d_model=d_model, d_ff=d_ff, n_qubits=n_qubits, n_layers=1, backend_name=backend_name, shots=shots)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        # Quantum Attention + Residual + LN
        attn_out = self.quantum_attn(x, mask=mask)
        x = self.norm1(x + attn_out)
        # Quantum FFN + Residual (handled internally in QuantumFeedForward)
        x = self.quantum_ffn(x)
        return self.head(x)


# ==========================================
# Model C: Compact Quantum-Native Transformer Research Prototype
# ==========================================
class QuantumNativeTransformer(nn.Module):
    """
    Model C: Compact Quantum-Native Transformer Research Prototype
    Small-register autoregressive model with parallel VQC heads and measurement-derived token probabilities.
    Directly routes sequence tokens into qubit state-space and generates next-token probabilities from quantum measurements.
    """

    def __init__(
        self,
        vocab_size: int = 8,
        n_qubits: int = 3,
        n_layers: int = 2,
        max_seq_len: int = 8,
        backend_name: str = "default.qubit",
        shots: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = min(vocab_size, 2 ** n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token embedding into qubit rotation angles
        self.token_embed = nn.Embedding(vocab_size, n_qubits)

        # Trainable quantum circuit weights
        self.vqc_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.05)

        # Context integration matrix
        self.context_gate = nn.Linear(n_qubits, n_qubits)

        self.device = DeviceManager.create_device(backend_name, n_qubits=n_qubits, shots=shots)
        wires = list(range(n_qubits))

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def _circuit(token_angles: torch.Tensor, weights: torch.Tensor):
            for i in range(n_qubits):
                qml.RY(token_angles[i], wires=wires[i])

            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=wires[i])
                    qml.RZ(weights[l, i, 1], wires=wires[i])
                for i in range(n_qubits):
                    qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])

            # Output computational basis probabilities (2^n states)
            return qml.probs(wires=wires)

        self._circuit = _circuit

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        angles = self.token_embed(input_ids)  # [batch_size, seq_len, n_qubits]

        logits_list = []
        for b in range(batch_size):
            seq_logits = []
            context = torch.zeros(self.n_qubits, dtype=angles.dtype, device=angles.device)
            for t in range(seq_len):
                # Context integration
                token_state = angles[b, t] + torch.tanh(self.context_gate(context))
                norm_state = normalize_quantum_features(token_state)
                # Run VQC
                probs = self._circuit(norm_state, self.vqc_weights)
                probs_t = torch.stack(probs) if isinstance(probs, (list, tuple)) else probs
                # Truncate or map to vocab_size
                token_probs = probs_t[:self.vocab_size].to(dtype=angles.dtype)
                # Compute log-probabilities for logits
                logits = torch.log(torch.clamp(token_probs, min=1e-6, max=1.0))
                seq_logits.append(logits)
                # Update context
                context = context + token_state
            logits_list.append(torch.stack(seq_logits))

        return torch.stack(logits_list)
