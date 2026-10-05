"""Unit tests for Paddle Quantum, CUDA-Q, Microsoft Quantum bridges and Subordinate Hybrid Layer."""

import pytest
import torch
import torch.nn as nn

from voidformer.quantum import (
    UnifiedQuantumBackendRegistry,
    PaddleQuantumBridge,
    CUDAQuantumBridge,
    MicrosoftQuantumBridge,
    UnifiedQuantumLayer,
    SubordinateHybridLayer,
    DataReuploadingVQCLayer,
    QuantumFourierTransformMap,
)
from voidformer.models.quantum_voidformer import QuantumVoidFormer
from voidformer.harness.deepseek_quantum_harness import DeepSeekQuantumHarness


def test_unified_backend_registry():
    backends = UnifiedQuantumBackendRegistry.get_available_backends()
    assert "pytorch_virtual" in backends
    assert backends["pytorch_virtual"] is True

    resolved = UnifiedQuantumBackendRegistry.resolve_backend("non_existent_backend")
    assert resolved == "pytorch_virtual"


def test_paddle_quantum_bridge():
    bridge = PaddleQuantumBridge(n_qubits=4)
    x = torch.randn(2, 4, 16)
    out = bridge.execute_pqc(x)
    assert out.shape == x.shape


def test_cuda_quantum_bridge():
    bridge = CUDAQuantumBridge(n_qubits=4)
    x = torch.randn(2, 4, 4)
    out = bridge.execute_kernel(x)
    assert out.shape == x.shape


def test_microsoft_quantum_bridge():
    bridge = MicrosoftQuantumBridge(n_qubits=4)
    res = bridge.estimate_resources(depth=20, gate_counts={"H": 8, "CNOT": 12, "T": 4})
    assert "logical_qubits" in res
    assert res["logical_qubits"] == 4
    assert res["t_count"] == 4


def test_unified_quantum_layer():
    layer = UnifiedQuantumLayer(d_model=16, n_qubits=4, backend="pytorch_virtual")
    x = torch.randn(2, 6, 16)
    out, diag = layer(x)
    assert out.shape == x.shape
    assert diag["backend"] == "pytorch_virtual"


def test_subordinate_hybrid_layer():
    hybrid_layer = SubordinateHybridLayer(d_model=16, n_qubits=4)
    x = torch.randn(2, 6, 16)
    out, diag = hybrid_layer(x, force_equal_split=True)
    assert out.shape == x.shape
    assert diag["quantum_workload_percentage"] == 50.0
    assert diag["classical_workload_percentage"] == 50.0


def test_data_reuploading_layer():
    reupload = DataReuploadingVQCLayer(d_model=16, n_qubits=4, reupload_layers=2)
    x = torch.randn(2, 6, 16)
    out, diag = reupload(x)
    assert out.shape == x.shape
    assert diag["reupload_layers"] == 2


def test_qft_feature_map():
    qft = QuantumFourierTransformMap(d_model=16, n_qubits=4)
    x = torch.randn(2, 6, 16)
    out = qft(x)
    assert out.shape == x.shape


def test_quantum_voidformer_full_forward():
    model = QuantumVoidFormer(
        vocab_size=50,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        n_qubits_per_token=4,
    )
    ids = torch.randint(0, 50, (2, 8))
    out = model(ids, return_diagnostics=True)
    assert out.logits.shape == (2, 8, 50)
    assert len(out.layer_diagnostics) == 2
    assert "subordinate_hybrid" in out.layer_diagnostics[0]


def test_deepseek_quantum_harness():
    model = QuantumVoidFormer(vocab_size=50, d_model=16, n_layers=1)
    harness = DeepSeekQuantumHarness(d_model=16, n_vqc_qubits=4, group_size=2)
    ids = torch.randint(0, 50, (2, 6))
    logits, diag = harness.evaluate_reasoning_task(model, ids)
    assert logits.shape == (2, 6, 50)
    assert diag["group_size"] == 2
    assert "resource_estimation" in diag
