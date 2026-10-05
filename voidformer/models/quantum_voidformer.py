"""Quantum-Enhanced VoidFormer Model.

Integrates the Virtual Quantum Computing Simulator into the VoidFormer architecture.
Replaces classical dual-state processing with quantum superposition, entanglement,
and measurement-based collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from voidformer.quantum.thermal_sampler import QuantumBoltzmannSampler
from voidformer.quantum.superposition_thinking import QuantumSuperpositionReasoningEngine
from voidformer.quantum.superposition_moe import QuantumSuperpositionMoE
from voidformer.quantum import (
    SubordinateHybridLayer,
    DataReuploadingVQCLayer,
    QuantumFourierTransformMap,
    QuantumSelfImprovingEngine,
)


@dataclass
class QuantumVoidFormerOutput:
    """Output from quantum-enhanced VoidFormer."""
    logits: torch.Tensor                        # (B, T, vocab_size)
    quantum_states: torch.Tensor                # Final quantum state info
    classical_output: torch.Tensor              # Collapsed classical states
    quantum_diagnostics: dict = field(default_factory=dict)
    layer_diagnostics: list[dict] = field(default_factory=list)


class QuantumVoidFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_qubits: int = 4,
        n_vqc_qubits: int = 4,
        n_vqc_layers: int = 2,
        use_vqc_layer: bool = True,
        vqc_backend: str = "statevector",
        vqc_shots: int = 1024,
        dropout: float = 0.1,
        use_quantum_attention: bool = True,
        use_tensor_network_ffn: bool = False,
        use_superposition_thinking: bool = True,
        thinking_steps: int = 4,
        use_quantum_moe: bool = False,
        num_experts: int = 4,
        top_k_experts: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.use_superposition_thinking = use_superposition_thinking
        self.use_quantum_moe = use_quantum_moe

        if use_superposition_thinking:
            self.qsre = QuantumSuperpositionReasoningEngine(
                d_model=d_model, n_vqc_qubits=n_vqc_qubits, thinking_steps=thinking_steps
            )

        from voidformer.layers.void_attention import VoidAttention
        self.attention = VoidAttention(d_void=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Quantum layers
        self.subordinate_hybrid = SubordinateHybridLayer(d_model=d_model, n_qubits=n_qubits)
        self.data_reuploading = DataReuploadingVQCLayer(d_model=d_model, n_qubits=n_qubits)
        self.qft_map = QuantumFourierTransformMap(d_model=d_model, n_qubits=n_qubits)
        self.self_improving_engine = QuantumSelfImprovingEngine(d_model=d_model, n_qubits=n_qubits)

        if use_quantum_moe:
            self.quantum_moe = QuantumSuperpositionMoE(
                d_model=d_model, n_vqc_qubits=n_vqc_qubits, num_experts=num_experts, top_k_experts=top_k_experts
            )

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        diagnostics = {}
        if self.use_superposition_thinking and hasattr(self, "qsre"):
            x = self.qsre(x)
            diagnostics["superposition_thinking"] = True

        attn_out, _, entropy = self.attention(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Execute subordinate hybrid layer
        x, sub_diag = self.subordinate_hybrid(x)
        diagnostics["subordinate_hybrid"] = True

        # Execute data reuploading and qft
        x, reup_diag = self.data_reuploading(x)
        x = self.qft_map(x)

        # Execute self-improving engine
        x, self_imp_diag = self.self_improving_engine(x)
        diagnostics["self_improving_engine"] = True

        if self.use_quantum_moe and hasattr(self, "quantum_moe"):
            x = self.quantum_moe(x)
            diagnostics["quantum_moe"] = True

        diagnostics["entropy"] = entropy
        return x, diagnostics


class QuantumVoidFormer(nn.Module):
    """Quantum-Enhanced VoidFormer Language Model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        n_qubits_per_token: int = 4,
        n_vqc_qubits: int = 4,
        n_vqc_layers: int = 2,
        use_vqc_layer: bool = True,
        vqc_backend: str = "statevector",
        vqc_shots: int = 1024,
        max_seq_len: int = 512,
        collapse_protocol: str = "entropy_gated",
        enable_entanglement: bool = True,
        use_quantum_attention: bool = True,
        use_tensor_network_ffn: bool = False,
        use_superposition_thinking: bool = True,
        thinking_steps: int = 4,
        use_quantum_moe: bool = False,
        num_experts: int = 4,
        top_k_experts: int = 2,
        use_quantum_token_embedder: bool = False,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            QuantumVoidFormerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_qubits=n_qubits_per_token,
                n_vqc_qubits=n_vqc_qubits,
                n_vqc_layers=n_vqc_layers,
                use_vqc_layer=use_vqc_layer,
                vqc_backend=vqc_backend,
                vqc_shots=vqc_shots,
                use_superposition_thinking=use_superposition_thinking,
                thinking_steps=thinking_steps,
                use_quantum_moe=use_quantum_moe,
                num_experts=num_experts,
                top_k_experts=top_k_experts,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        ids: torch.Tensor,
        use_quantum_processing: bool = True,
        quantum_algorithm: Optional[Any] = None,
        return_diagnostics: bool = False,
    ) -> QuantumVoidFormerOutput:
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.embedding(ids) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        layer_diagnostics = []
        for i, block in enumerate(self.blocks):
            x, block_diag = block(x, return_diagnostics=return_diagnostics)
            if return_diagnostics:
                block_diag["layer_idx"] = i
                layer_diagnostics.append(block_diag)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return QuantumVoidFormerOutput(
            logits=logits,
            quantum_states=x,
            classical_output=x,
            quantum_diagnostics={},
            layer_diagnostics=layer_diagnostics,
        )

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_quantum: bool = True,
    ) -> torch.Tensor:
        self.eval()
        sampler = QuantumBoltzmannSampler(temperature=temperature)

        for _ in range(max_new_tokens):
            ids_crop = ids[:, -self.max_seq_len:]
            out = self.forward(ids_crop, use_quantum_processing=use_quantum)
            logits = out.logits[:, -1, :]

            entropy_scale = 0.0
            if "measurement_entropy" in out.quantum_diagnostics:
                entropy = out.quantum_diagnostics["measurement_entropy"]
                entropy_scale = entropy.mean().item() if isinstance(entropy, torch.Tensor) else float(entropy)

            next_token = sampler.sample(logits, entropy_scale=entropy_scale, top_k=top_k)
            ids = torch.cat([ids, next_token], dim=1)

        return ids
