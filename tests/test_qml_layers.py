"""Unit tests for QML hybrid layers and networks."""

import pytest
import torch
from qml.layers import HybridNet, ClassicalNet
from qml.backends import get_pennylane_device
from qml.train import get_sentiment_dataset, train_and_eval
from torch.utils.data import TensorDataset, DataLoader


def test_pennylane_device():
    dev = get_pennylane_device("simulator", n_qubits=4)
    assert dev.wires.labels == (0, 1, 2, 3)


def test_hybrid_net_forward_and_backward():
    net = HybridNet(in_features=16, n_qubits=4, n_layers=2, num_classes=2)
    x = torch.randn(4, 16)
    out = net(x)
    assert out.shape == (4, 2)

    loss = out.sum()
    loss.backward()
    assert net.pre.weight.grad is not None
    assert net.post.weight.grad is not None


def test_classical_net_parameter_matching():
    h = HybridNet(in_features=16, n_qubits=4, n_layers=2, num_classes=2)
    c = ClassicalNet(in_features=16, hidden_dim=4, num_classes=2, match_parameters_of=h)

    x = torch.randn(4, 16)
    out = c(x)
    assert out.shape == (4, 2)

    h_params = sum(p.numel() for p in h.parameters() if p.requires_grad)
    c_params = sum(p.numel() for p in c.parameters() if p.requires_grad)
    assert abs(h_params - c_params) < 10


def test_synthetic_sentiment_dataset():
    X, y = get_sentiment_dataset(n_samples=20, n_components=16, seed=42, use_synthetic=True)
    assert X.shape == (20, 16)
    assert len(y) == 20


def test_train_and_eval_loop():
    X, y = get_sentiment_dataset(n_samples=20, n_components=16, seed=42, use_synthetic=True)
    train_ds = TensorDataset(torch.tensor(X[:10], dtype=torch.float32), torch.tensor(y[:10], dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X[10:], dtype=torch.float32), torch.tensor(y[10:], dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=5)
    val_loader = DataLoader(val_ds, batch_size=5)

    net = HybridNet(in_features=16, n_qubits=4, n_layers=1, num_classes=2)
    loss, acc = train_and_eval(net, train_loader, val_loader, epochs=1, lr=0.01)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0
