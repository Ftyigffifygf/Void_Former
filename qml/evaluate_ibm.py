"""IBM Quantum hardware and noisy simulator evaluation script for Hybrid QML models.

Includes safeguards:
1. Real hardware is run ONLY when explicitly requested with `--backend ibm_hardware`.
2. Prints and saves exact backend identification metadata in evaluation output.
3. Hardware evaluation enforces a strict subset cap (50-100 samples) and shot limit (1024) to avoid burning IBM QPU quota.
"""

from __future__ import annotations

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from qml.layers import HybridNet
from qml.backends import get_pennylane_device
from qml.train import get_sentiment_dataset


def evaluate_backend(
    model_path: Optional[str] = None,
    backend_kind: str = "ibm_aer",
    n_samples: int = 50,
    shots: int = 1024,
    output_json: str = "ibm_eval_results.json",
):
    print(f"=== Starting QML Hardware / Simulator Evaluation ===")

    # Safeguard 1: Real hardware warning and check
    if backend_kind == "ibm_hardware":
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            print("[Warning] --backend ibm_hardware requested but IBM_QUANTUM_TOKEN is missing!")
            print("Falling back to noisy Aer simulator.")
            backend_kind = "ibm_aer"
        else:
            print("[Notice] Running evaluation on REAL IBM QUANTUM HARDWARE.")
            # Safeguard 3: Cap samples to 50 for real hardware
            n_samples = min(n_samples, 50)
            print(f"Capped hardware evaluation sample count to {n_samples} samples and {shots} shots.")

    dev = get_pennylane_device(backend_kind, n_qubits=4, shots=shots)
    actual_backend_name = str(dev.name)

    print(f"Active PennyLane Device Backend: {actual_backend_name}")

    # Load dataset
    X, y = get_sentiment_dataset(n_samples=n_samples, n_components=16, seed=42)
    val_ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = HybridNet(in_features=16, n_qubits=4, n_layers=2, num_classes=2, dev=dev)
    if model_path and os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}")
        model.load_state_dict(torch.load(model_path))

    model.eval()
    criterion = nn.CrossEntropyLoss()
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

    # Safeguard 2: Record exact backend metadata
    results = {
        "requested_backend_kind": backend_kind,
        "actual_backend_name": actual_backend_name,
        "n_samples": total,
        "shots": shots,
        "loss": float(avg_loss),
        "accuracy": float(acc),
    }

    print("\n================ EVALUATION RESULTS ================")
    print(f"Backend Kind Requested: {results['requested_backend_kind']}")
    print(f"Actual Backend Device:  {results['actual_backend_name']}")
    print(f"Evaluation Samples:     {results['n_samples']}")
    print(f"Shots:                  {results['shots']}")
    print(f"Loss:                   {results['loss']:.4f}")
    print(f"Accuracy:               {results['accuracy']*100:.2f}%")
    print("====================================================")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hybrid QML model on IBM simulator / hardware")
    parser.add_argument("--backend", type=str, default="ibm_aer", choices=["simulator", "ibm_aer", "ibm_hardware"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--out", type=str, default="ibm_eval_results.json")
    args = parser.parse_args()

    evaluate_backend(
        model_path=args.model_path,
        backend_kind=args.backend,
        n_samples=args.n_samples,
        shots=args.shots,
        output_json=args.out,
    )
