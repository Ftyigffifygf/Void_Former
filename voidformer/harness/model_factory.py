"""Model Factory for Harness Package."""

from __future__ import annotations

import torch
import torch.nn as nn

from voidformer.models.quantum_voidformer import QuantumVoidFormer
from voidformer.models.voidformer import VoidFormerModel


def create_model(
    model_type: str = "quantum",
    vocab_size: int = 128,
    d_model: int = 64,
    d_void: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    d_ff: int = 256,
    max_seq_len: int = 128,
    collapse_protocol: str = "entropy_gated",
    enable_entanglement: bool = True,
    use_tensor_network_ffn: bool = False,
    n_qubits_per_token: int = 4,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """Create a VoidFormer or QuantumVoidFormer model instance."""
    dev = torch.device(device)

    if "seq_len" in kwargs:
        max_seq_len = kwargs.pop("seq_len")

    if model_type == "classical":
        return VoidFormerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            d_void=d_void,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
        ).to(dev)

    elif model_type == "quantum":
        use_vqc_layer = kwargs.get("use_vqc_layer", True)
        use_superposition_thinking = kwargs.get("use_superposition_thinking", True)
        thinking_steps = kwargs.get("thinking_steps", 4)
        use_quantum_moe = kwargs.get("use_quantum_moe", False)
        num_experts = kwargs.get("num_experts", 4)
        top_k_experts = kwargs.get("top_k_experts", 2)
        use_quantum_token_embedder = kwargs.get("use_quantum_token_embedder", False)

        return QuantumVoidFormer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            n_qubits_per_token=n_qubits_per_token,
            max_seq_len=max_seq_len,
            collapse_protocol=collapse_protocol,
            enable_entanglement=enable_entanglement,
            use_vqc_layer=use_vqc_layer,
            use_tensor_network_ffn=use_tensor_network_ffn,
            use_superposition_thinking=use_superposition_thinking,
            thinking_steps=thinking_steps,
            use_quantum_moe=use_quantum_moe,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            use_quantum_token_embedder=use_quantum_token_embedder,
            device=dev,
        ).to(dev)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
