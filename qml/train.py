"""Side-by-side training experiment for HybridNet vs ClassicalNet on text sentiment."""

from __future__ import annotations

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups

from qml.layers import HybridNet, ClassicalNet


def get_sentiment_dataset(
    n_samples: int = 1000,
    n_components: int = 16,
    seed: int = 42,
    use_synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate binary text dataset reduced to n_components features using TF-IDF + SVD."""
    if use_synthetic:
        random.seed(seed)
        pos_words = ["good", "great", "excellent", "awesome", "fantastic", "positive", "love", "wonderful"]
        neg_words = ["bad", "terrible", "horrible", "awful", "negative", "hate", "poor", "worst"]
        texts = []
        labels = []
        for i in range(n_samples):
            label = i % 2
            words = pos_words if label == 1 else neg_words
            text = " ".join(random.choices(words, k=10))
            texts.append(text)
            labels.append(label)
        labels = np.array(labels)
    else:
        try:
            data = fetch_20newsgroups(
                subset="all",
                categories=["sci.med", "sci.space"],
                remove=("headers", "footers", "quotes"),
            )
            texts = data.data[:n_samples]
            labels = data.target[:n_samples]
        except Exception:
            # Synthetic text fallback if fetch fails or offline
            random.seed(seed)
            pos_words = ["good", "great", "excellent", "awesome", "fantastic", "positive", "love", "wonderful"]
            neg_words = ["bad", "terrible", "horrible", "awful", "negative", "hate", "poor", "worst"]
            texts = []
            labels = []
            for i in range(n_samples):
                label = i % 2
                words = pos_words if label == 1 else neg_words
                text = " ".join(random.choices(words, k=10))
                texts.append(text)
                labels.append(label)
            labels = np.array(labels)

    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X_tfidf = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    X_svd = svd.fit_transform(X_tfidf)

    return X_svd, labels


def train_and_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.01,
) -> tuple[float, float]:
    """Train model and return validation loss and accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in val_loader:
            out = model(bx)
            loss = criterion(out, by)
            val_loss += loss.item() * bx.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

    avg_loss = val_loss / total
    acc = correct / total
    return avg_loss, acc


def run_experiment(
    seeds: list[int] = [42, 43, 44],
    n_samples: int = 1000,
    epochs: int = 15,
    lr: float = 0.01,
    batch_size: int = 32,
    use_synthetic: bool = False,
    output_json: str = "qml_benchmark_results.json",
):
    hybrid_losses, hybrid_accs = [], []
    classical_losses, classical_accs = [], []

    print(f"=== Starting Hybrid QML vs Classical Baseline Experiment ({len(seeds)} seeds) ===")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        X, y = get_sentiment_dataset(n_samples=n_samples, n_components=16, seed=seed, use_synthetic=use_synthetic)

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Hybrid model
        hybrid_net = HybridNet(in_features=16, n_qubits=4, n_layers=2, num_classes=2)
        h_loss, h_acc = train_and_eval(hybrid_net, train_loader, val_loader, epochs=epochs, lr=lr)
        hybrid_losses.append(h_loss)
        hybrid_accs.append(h_acc)

        # Classical baseline
        classical_net = ClassicalNet(in_features=16, hidden_dim=4, num_classes=2, match_parameters_of=hybrid_net)
        c_loss, c_acc = train_and_eval(classical_net, train_loader, val_loader, epochs=epochs, lr=lr)
        classical_losses.append(c_loss)
        classical_accs.append(c_acc)

        print(f"Seed {seed}: Hybrid Acc = {h_acc*100:.2f}%, Loss = {h_loss:.4f} | Classical Acc = {c_acc*100:.2f}%, Loss = {c_loss:.4f}")

    results = {
        "seeds": seeds,
        "hybrid": {
            "acc_mean": float(np.mean(hybrid_accs)),
            "acc_std": float(np.std(hybrid_accs)),
            "loss_mean": float(np.mean(hybrid_losses)),
            "loss_std": float(np.std(hybrid_losses)),
            "accs": hybrid_accs,
            "losses": hybrid_losses,
        },
        "classical": {
            "acc_mean": float(np.mean(classical_accs)),
            "acc_std": float(np.std(classical_accs)),
            "loss_mean": float(np.mean(classical_losses)),
            "loss_std": float(np.std(classical_losses)),
            "accs": classical_accs,
            "losses": classical_losses,
        },
    }

    print("\n================ FINAL COMPARISON SUMMARY ================")
    print(f"HybridNet Accuracy:    {results['hybrid']['acc_mean']*100:.2f}% ± {results['hybrid']['acc_std']*100:.2f}%")
    print(f"HybridNet Loss:        {results['hybrid']['loss_mean']:.4f} ± {results['hybrid']['loss_std']:.4f}")
    print(f"ClassicalNet Accuracy: {results['classical']['acc_mean']*100:.2f}% ± {results['classical']['acc_std']*100:.2f}%")
    print(f"ClassicalNet Loss:     {results['classical']['loss_mean']:.4f} ± {results['classical']['loss_std']:.4f}")
    print("==========================================================")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid QML model vs Classical Baseline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic dataset instead of downloading 20newsgroups")
    parser.add_argument("--out", type=str, default="qml_benchmark_results.json")
    args = parser.parse_args()

    run_experiment(
        seeds=args.seeds,
        n_samples=args.n_samples,
        epochs=args.epochs,
        use_synthetic=args.use_synthetic,
        output_json=args.out,
    )
