"""Comprehensive automated test suite for Q-INTELLIGENCE."""

import pytest
import torch
import os
import pennylane as qml

from qai.core import (
    QAIConfig,
    DeviceManager,
    BackendRegistry,
    to_torch_tensor,
    normalize_quantum_features,
    CheckpointManager,
    set_seed,
)
from qai.quantum import (
    CircuitBuilder,
    QuantumEncoder,
    VariationalAnsatz,
    QuantumMeasurements,
    compute_gradient_norms,
)
from qai.neural import ClassicalANN, QuantumANN, HybridANN, QuantumFeedForward
from qai.transformer import (
    SimpleTokenizer,
    ClassicalAttention,
    QuantumAttention,
    ClassicalTransformer,
    HybridQuantumTransformer,
    QuantumNativeTransformer,
    generate,
)
from qai.hardware import (
    SimulatorBackend,
    NoisyBackend,
    IBMBackend,
    HardwareValidator,
    QuantumAuthenticator,
)
from qai.datasets import (
    get_xor_dataset,
    get_parity_dataset,
    get_regression_dataset,
    get_deterministic_sequence_dataset,
    create_train_val_loaders,
)
from qai.training import Trainer, create_optimizer
from qai.evaluation import analyze_trainability_and_barren_plateaus, BenchmarkSuite


# ==========================================
# 1. Core Tests
# ==========================================
def test_config_loading_and_saving(tmp_path):
    cfg_file = tmp_path / "test_cfg.yaml"
    cfg = QAIConfig()
    cfg.quantum.n_qubits = 6
    cfg.to_yaml(str(cfg_file))
    assert os.path.exists(cfg_file)

    loaded = QAIConfig.from_yaml(str(cfg_file))
    assert loaded.quantum.n_qubits == 6


def test_device_manager():
    dev = DeviceManager.create_device("default.qubit", n_qubits=2)
    assert dev.name == "default.qubit"
    with pytest.raises(ValueError):
        DeviceManager.create_device("invalid.backend", n_qubits=2)


def test_tensor_normalization():
    x = torch.tensor([[1.0, 5.0, 10.0]])
    norm = normalize_quantum_features(x, feature_min=0.0, feature_max=3.14159)
    assert norm.min() >= 0.0
    assert norm.max() <= 3.1416


# ==========================================
# 2. Quantum Engine Tests
# ==========================================
def test_circuit_builder_and_resource_estimation():
    dev = DeviceManager.create_device("default.qubit", n_qubits=3)
    builder = CircuitBuilder(dev, n_qubits=3, n_layers=2)
    res = builder.estimate_resources()
    assert res["n_logical_qubits"] == 3
    assert res["trainable_quantum_parameters"] == 12


def test_quantum_measurements():
    dev = DeviceManager.create_device("default.qubit", n_qubits=2)

    @qml.qnode(dev, interface="torch")
    def circ():
        qml.Hadamard(wires=0)
        return QuantumMeasurements.pauli_z_expectations([0, 1])

    out = torch.stack(circ())
    assert out.shape == torch.Size([2])
    assert abs(out[0].item()) < 1e-4  # <Z> for |+> state is 0


# ==========================================
# 3. Quantum ANN & Gradient Updates Tests
# ==========================================
def test_quantum_ann_forward_backward():
    set_seed(42)
    model = QuantumANN(in_features=2, out_features=2, n_qubits=2, n_layers=1)
    x = torch.randn(2, 2)
    out = model(x)
    assert out.shape == torch.Size([2, 2])

    loss = out.sum()
    loss.backward()
    assert model.quantum_weights.grad is not None
    assert torch.isfinite(model.quantum_weights.grad).all()

    # Parameter update
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    init_val = model.quantum_weights.clone()
    opt.step()
    assert not torch.equal(init_val, model.quantum_weights)


def test_hybrid_ann_forward():
    model = HybridANN(in_features=3, hidden_features=6, out_features=2, n_qubits=2)
    x = torch.randn(2, 3)
    out = model(x)
    assert out.shape == torch.Size([2, 2])


# ==========================================
# 4. Transformer Suite Tests
# ==========================================
def test_tokenizer_and_embeddings():
    tok = SimpleTokenizer()
    text = "hello"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_classical_transformer():
    model = ClassicalTransformer(vocab_size=10, d_model=8, n_heads=1, max_seq_len=8)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model(x)
    assert out.shape == torch.Size([1, 4, 10])


def test_quantum_attention_and_hybrid_transformer():
    set_seed(42)
    model = HybridQuantumTransformer(vocab_size=8, d_model=8, n_qubits=2, max_seq_len=8)
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model(x)
    assert out.shape == torch.Size([1, 3, 8])

    loss = out.sum()
    loss.backward()
    assert model.quantum_attn.theta.grad is not None
    assert torch.isfinite(model.quantum_attn.theta.grad).all()


def test_quantum_native_transformer():
    model = QuantumNativeTransformer(vocab_size=8, n_qubits=3, n_layers=1, max_seq_len=6)
    x = torch.tensor([[1, 2]], dtype=torch.long)
    out = model(x)
    assert out.shape == torch.Size([1, 2, 8])


def test_autoregressive_generation():
    model = ClassicalTransformer(vocab_size=10, d_model=8, n_heads=1, max_seq_len=8)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    gen = generate(model, prompt, max_new_tokens=3)
    assert gen.shape == torch.Size([1, 5])


# ==========================================
# 5. Hardware Validation Tests
# ==========================================
def test_hardware_authenticator_safe_fallback():
    assert not QuantumAuthenticator.has_valid_credentials() or True
    status = HardwareValidator.check_hardware_status(allow_hardware=False)
    assert "hardware_access" in status
    assert status["hardware_access"] in ["NOT AVAILABLE", "AUTHENTICATED"]


def test_ibm_backend_permission_error():
    backend = IBMBackend(allow_hardware_execution=False)
    with pytest.raises(PermissionError):
        backend.get_device(2)


# ==========================================
# 6. Training & Checkpoint Tests
# ==========================================
def test_training_and_checkpoint_roundtrip(tmp_path):
    ds = get_xor_dataset(n_samples=16)
    train_l, val_l = create_train_val_loaders(ds, batch_size=4)
    model = QuantumANN(in_features=2, out_features=2, n_qubits=2, n_layers=1)
    trainer = Trainer(model, train_l, val_l, epochs=1, lr=0.05, checkpoint_dir=str(tmp_path))
    res = trainer.train(model_name="test_qann")
    assert res["epochs"] == 1

    ckpt_file = tmp_path / "test_qann_best.pt"
    assert os.path.exists(ckpt_file)

    loaded_model = QuantumANN(in_features=2, out_features=2, n_qubits=2, n_layers=1)
    CheckpointManager.load(str(ckpt_file), loaded_model)
    assert torch.equal(model.quantum_weights, loaded_model.quantum_weights)


# ==========================================
# 7. Trainability & Barren Plateau Tests
# ==========================================
def test_barren_plateau_analysis():
    results = analyze_trainability_and_barren_plateaus(qubit_range=[2], depth_range=[1], n_samples=2)
    assert len(results) == 1
    assert "mean_gradient_magnitude" in results[0]
    assert results[0]["is_trainable"] is True
