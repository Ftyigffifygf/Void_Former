"""Quantum-Enhanced VoidFormer Model.

Integrates the Virtual Quantum Computing Simulator into the VoidFormer architecture.
Replaces classical dual-state processing with quantum superposition, entanglement,
and measurement-based collapse.

ARCHITECTURE TRANSFORMATION:
    Classical VoidFormer: |T⟩ = α|S_c⟩ + β|S_v⟩ + γ·I(|S_c⟩,|S_v⟩)
    ↓
    Quantum VoidFormer: |ψ⟩ = Σᵢ αᵢ|i⟩ in 2^n Hilbert space

where quantum superposition replaces dual-state representation,
entanglement replaces interference, and measurement replaces collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

try:
    from ..quantum import (
        VirtualQuantumProcessor,
        QuantumCircuit,
        QuantumAlgorithm,
        CollapseProtocol,
        QuantumKernelAttention,
        QuantumInspiredNeuralLayer,
        VQCLayer,
        QuantumSuperpositionReasoningEngine,
        QuantumSuperpositionTokenEmbedder,
        QuantumSuperpositionMoE,
    )
    from ..layers import DualEmbedding
except (ImportError, ValueError):
    from voidformer.quantum import (
        QuantumAlgorithm,
        QuantumKernelAttention,
        QuantumInspiredNeuralLayer,
        VQCLayer,
        QuantumSuperpositionReasoningEngine,
        QuantumSuperpositionTokenEmbedder,
        QuantumSuperpositionMoE,
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
    """Single transformer block with quantum processing.

    Replaces classical attention and FFN with quantum operations:
    1. Quantum Kernel Attention (quantum fidelity-based attention)
    2. Quantum Circuit Processing (gate operations)
    3. Quantum-Inspired FFN (tensor network or QIML layer)
    """

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
        self.use_quantum_attention = use_quantum_attention
        self.use_superposition_thinking = use_superposition_thinking
        self.use_quantum_moe = use_quantum_moe

        # Quantum or classical attention
        if use_quantum_attention:
            self.attention = QuantumKernelAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_qubits=n_qubits,
                dropout=dropout,
            )
        else:
            from ..layers.classical_attention import ClassicalAttention
            self.attention = ClassicalAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )

        self.norm1 = nn.LayerNorm(d_model)

        # Quantum-inspired FFN or classical FFN
        if use_tensor_network_ffn:
            from ..quantum import TensorNetworkLayer
            self.ffn = TensorNetworkLayer(
                d_in=d_model,
                d_out=d_model,
                bond_dim=min(16, d_model // 8),
                n_cores=4,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        self.norm2 = nn.LayerNorm(d_model)

        # Optional: Quantum-inspired neural layer
        self.quantum_layer = QuantumInspiredNeuralLayer(
            d_model=d_model,
            n_qubits=n_qubits,
            use_entanglement=True,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Subordinate Hybrid Layer (50% Quantum / 50% Classical Workload Split)
        from ..quantum import SubordinateHybridLayer, DataReuploadingVQCLayer, QuantumFourierTransformMap
        self.subordinate_hybrid = SubordinateHybridLayer(
            d_model=d_model,
            n_qubits=n_qubits,
            backend=vqc_backend if vqc_backend in ["paddle_quantum", "cudaq", "microsoft_quantum"] else "pytorch_virtual",
            dropout=dropout,
        )

        # Data Re-uploading QNN Layer
        self.data_reuploading = DataReuploadingVQCLayer(
            d_model=d_model,
            n_qubits=n_qubits,
            reupload_layers=2,
            dropout=dropout,
        )

        # Quantum Fourier Transform Feature Map
        self.qft_map = QuantumFourierTransformMap(
            d_model=d_model,
            n_qubits=n_qubits,
        )

        # Self-Improving Quantum Engine
        from ..quantum import QuantumSelfImprovingEngine
        self.self_improving_engine = QuantumSelfImprovingEngine(
            d_model=d_model,
            n_qubits=n_qubits,
        )

        # Hardware-bound VQC layer
        self.use_vqc_layer = use_vqc_layer
        if use_vqc_layer:
            self.vqc_layer = VQCLayer(
                d_model=d_model,
                n_vqc_qubits=n_vqc_qubits,
                n_vqc_layers=n_vqc_layers,
                backend=vqc_backend if vqc_backend in ["aer", "statevector"] else "statevector",
                shots=vqc_shots,
                dropout=dropout,
            )

        # Quantum Superposition Reasoning Engine (QSRE)
        if use_superposition_thinking:
            self.qsre = QuantumSuperpositionReasoningEngine(
                d_model=d_model,
                n_vqc_qubits=n_vqc_qubits,
                thinking_steps=thinking_steps,
            )
            self.superposition_engine = self.qsre

        # Quantum Superposition Mixture of Experts (MoE)
        if use_quantum_moe:
            self.quantum_moe = QuantumSuperpositionMoE(
                d_model=d_model,
                n_vqc_qubits=n_vqc_qubits,
                num_experts=num_experts,
                top_k_experts=top_k_experts,
                thinking_steps=thinking_steps,
            )

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, T, d_model)
            return_diagnostics: Whether to return quantum diagnostics

        Returns:
            output: (B, T, d_model)
            diagnostics: Dict of processing statistics
        """
        diagnostics = {}

        # 0. Quantum Superposition Reasoning Engine (QSRE - prior to attention/FFN)
        if self.use_superposition_thinking and hasattr(self, "qsre"):
            x = self.qsre(x)
            diagnostics["superposition_thinking"] = True

        # 1. Quantum attention
        if self.use_quantum_attention:
            attn_out, attn_diag = self.attention(x)
            diagnostics["attention"] = attn_diag
        else:
            attn_out, _ = self.attention(x)

        x = self.norm1(x + attn_out)

        # 2. FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # 3. Subordinate Hybrid Processing (50% Quantum / 50% Classical Workload)
        x, sub_diag = self.subordinate_hybrid(x)
        diagnostics["subordinate_hybrid"] = sub_diag

        # 4. Data Re-uploading QNN Layer
        x, reup_diag = self.data_reuploading(x)
        diagnostics["data_reuploading"] = reup_diag

        # 5. QFT Feature Map Layer
        x = self.qft_map(x)

        # 6. Self-Improving Engine Parameter Adaptation
        x, self_imp_diag = self.self_improving_engine(x, fidelity_score=0.9)
        diagnostics["self_improving_engine"] = self_imp_diag

        # 7. Quantum-inspired processing layer
        quantum_out, quantum_diag = self.quantum_layer(x)
        diagnostics["quantum_layer"] = quantum_diag
        x = self.norm3(x + quantum_out)

        # 7. Hardware-bound VQC layer
        if self.use_vqc_layer:
            x, vqc_diag = self.vqc_layer(x)
            diagnostics["vqc_layer"] = vqc_diag

        # 5. Quantum Superposition Mixture of Experts
        if self.use_quantum_moe:
            x = self.quantum_moe(x)
            diagnostics["quantum_moe"] = True

        return x, diagnostics


class QuantumVoidFormer(nn.Module):
    """Quantum-Enhanced VoidFormer: Virtual Quantum Computing Simulator + LM.

    Replaces classical transformer with quantum computing simulation:
    - Token embeddings → Quantum superposition states
    - Attention → Quantum kernel attention (fidelity-based)
    - Processing → Quantum circuits with gates and entanglement
    - Output → Measurement-based collapse to classical logits

    The model operates primarily in quantum superposition during processing,
    only collapsing to classical outputs at the final measurement layer.
    """

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
        self.n_qubits = n_qubits_per_token
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.use_quantum_token_embedder = use_quantum_token_embedder

        # Token embeddings (classical → quantum via processor)
        if use_quantum_token_embedder:
            self.quantum_token_embedder = QuantumSuperpositionTokenEmbedder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_vqc_qubits=n_vqc_qubits,
            )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Virtual Quantum Processor (core quantum computing engine)
        try:
            from quantum_init import initialize_quantum_processor
        except ImportError:
            from ..quantum_init import initialize_quantum_processor

        self.quantum_processor = initialize_quantum_processor(
            d_model=d_model,
            n_qubits_per_token=n_qubits_per_token,
            max_seq_len=max_seq_len,
            collapse_protocol=collapse_protocol,
            enable_entanglement=enable_entanglement,
            device=device,
            name=f"qvf_{id(self)}",
        )

        # Quantum-enhanced transformer blocks
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
                dropout=dropout,
                use_quantum_attention=use_quantum_attention,
                use_tensor_network_ffn=use_tensor_network_ffn,
                use_superposition_thinking=use_superposition_thinking,
                thinking_steps=thinking_steps,
                use_quantum_moe=use_quantum_moe,
                num_experts=num_experts,
                top_k_experts=top_k_experts,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.01)
        if not hasattr(self.lm_head, '_is_tied'):
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self,
        ids: torch.Tensor,
        use_quantum_processing: bool = True,
        quantum_algorithm: Optional[QuantumAlgorithm] = None,
        return_diagnostics: bool = False,
    ) -> QuantumVoidFormerOutput:
        """
        Args:
            ids: Token IDs (B, T)
            use_quantum_processing: Whether to use quantum processor
            quantum_algorithm: Optional specific quantum algorithm to execute
            return_diagnostics: Return detailed quantum diagnostics

        Returns:
            QuantumVoidFormerOutput with logits and quantum information
        """
        B, T = ids.shape
        assert T <= self.max_seq_len

        # 1. Classical or Quantum Token Embeddings
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        if self.use_quantum_token_embedder and hasattr(self, "quantum_token_embedder"):
            x_quantum_emb, _ = self.quantum_token_embedder(ids)
            x = x_quantum_emb + self.pos_embedding(positions)
        else:
            x = self.embedding(ids) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        layer_diagnostics = []

        # 2. Process through quantum-enhanced blocks
        for i, block in enumerate(self.blocks):
            x, block_diag = block(x, return_diagnostics=return_diagnostics)
            if return_diagnostics:
                block_diag["layer_idx"] = i
                layer_diagnostics.append(block_diag)

        # 3. Pre-LM normalization
        x = self.ln_f(x)

        # 4. Quantum processing pipeline (optional)
        quantum_diagnostics = {}
        if use_quantum_processing:
            if quantum_algorithm is not None:
                # Execute specific quantum algorithm
                x_quantum, q_diag = self.quantum_processor.execute_algorithm(
                    quantum_algorithm, x
                )
            else:
                # Standard quantum circuit processing
                x_quantum, q_diag = self.quantum_processor(x)

            quantum_diagnostics = q_diag

            # Blend quantum and classical (hybrid mode)
            x = 0.5 * x + 0.5 * x_quantum

        # 5. Language modeling head
        logits = self.lm_head(x)

        return QuantumVoidFormerOutput(
            logits=logits,
            quantum_states=x,
            classical_output=x,
            quantum_diagnostics=quantum_diagnostics,
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
        """Generate tokens using quantum-enhanced sampling.

        Args:
            ids: Starting token IDs (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            use_quantum: Use quantum processing during generation

        Returns:
            Generated token sequences (B, T+max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max context
            ids_crop = ids[:, -self.max_seq_len:]

            # Forward pass with quantum
            out = self.forward(ids_crop, use_quantum_processing=use_quantum)
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)

            # Apply quantum measurement uncertainty to logits (optional)
            if use_quantum and "measurement_entropy" in out.quantum_diagnostics:
                entropy = out.quantum_diagnostics["measurement_entropy"]
                if isinstance(entropy, torch.Tensor):
                    entropy_scale = entropy.mean().item()
                else:
                    entropy_scale = entropy
                # Higher entropy → more uncertain → flatten distribution
                logits = logits / (1.0 + 0.1 * entropy_scale)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token], dim=1)

        return ids

    def num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_quantum_diagnostics(self, ids: torch.Tensor) -> dict:
        """Get detailed quantum state diagnostics without full forward pass."""
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.embedding(ids) + self.pos_embedding(positions)

        return self.quantum_processor.get_quantum_state_diagnostics(x)
