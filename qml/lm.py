"""Small-scale Character/Vocabulary Transformer LMs: Classical Baseline vs Hybrid Quantum LM."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import pennylane as qml

from qml.backends import get_pennylane_device


def create_vqc_ffn_circuit(n_qubits: int = 4, n_layers: int = 2, dev: Optional[qml.Device] = None):
    """PennyLane QNode for Variational Quantum Circuit (VQC) FFN block."""
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    diff_method = "parameter-shift" if (hasattr(dev, "shots") and dev.shots) else "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumFFNBlock(nn.Module):
    """Feed-Forward Block replaced with a 4-8 qubit Variational Quantum Circuit (VQC).

    Architecture:
      Classical Linear (d_model -> n_qubits) + Tanh
      PennyLane TorchLayer (n_qubits, StronglyEntanglingLayers) -> Pauli-Z expvals
      Classical Linear (n_qubits -> d_model)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend_kind: str = "simulator",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits

        dev = get_pennylane_device(backend_kind, n_qubits=n_qubits)
        self.qnode = create_vqc_ffn_circuit(n_qubits=n_qubits, n_layers=n_layers, dev=dev)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        self.down_proj = nn.Linear(d_model, n_qubits)
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.up_proj = nn.Linear(n_qubits, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, d_model)
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        h = torch.tanh(self.down_proj(x_flat))
        q_out = self.q_layer(h)
        out_flat = self.up_proj(q_out)
        return out_flat.reshape(B, T, D)


class ClassicalTransformerLM(nn.Module):
    """Small Classical Character/Small-Vocab Language Model (100K - 1M parameters)."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        x = self.dropout(x)

        mask = self._generate_causal_mask(T, input_ids.device)
        h = self.transformer(x, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


class HybridQuantumTransformerBlock(nn.Module):
    """Transformer block with FFN replaced by QuantumFFNBlock."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 2,
        n_qubits: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        backend_kind: str = "simulator",
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.q_ffn = QuantumFFNBlock(d_model=d_model, n_qubits=n_qubits, n_layers=n_layers, backend_kind=backend_kind)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.q_ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class HybridQuantumTransformerLM(nn.Module):
    """Hybrid Quantum Transformer Language Model with VQC FFN blocks."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        n_qubits: int = 4,
        n_vqc_layers: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        backend_kind: str = "simulator",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            HybridQuantumTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_qubits=n_qubits,
                n_layers=n_vqc_layers,
                dropout=dropout,
                backend_kind=backend_kind,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        x = self.dropout(x)

        mask = self._generate_causal_mask(T, input_ids.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_f(x)
        return self.head(x)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity PPL = exp(cross_entropy_loss)."""
    return math.exp(min(loss, 20.0))
