"""Training and Perplexity Benchmarking Script for Classical vs Hybrid Quantum LM."""

from __future__ import annotations

import argparse
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from qml.lm import ClassicalTransformerLM, HybridQuantumTransformerLM, compute_perplexity


SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause.
"""


class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text: str, seq_len: int = 64):
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len = seq_len

        encoded = [self.char2idx[ch] for ch in text]
        self.data = torch.tensor(encoded, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def train_lm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.003,
) -> tuple[float, float, float]:
    """Train language model and return val_loss, val_perplexity, train_loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits.view(-1, logits.size(-1)), by.view(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for bx, by in val_loader:
            logits = model(bx)
            loss = criterion(logits.view(-1, logits.size(-1)), by.view(-1))
            val_loss += loss.item() * by.numel()
            total_tokens += by.numel()

    avg_val_loss = val_loss / total_tokens
    val_ppl = compute_perplexity(avg_val_loss)
    return avg_val_loss, val_ppl, loss.item()


def run_lm_benchmark(
    text: str = SAMPLE_TEXT,
    seq_len: int = 32,
    epochs: int = 5,
    seeds: list[int] = [42, 43, 44],
    output_json: str = "qml_lm_benchmark_results.json",
):
    print(f"=== Starting Classical LM vs Hybrid Quantum LM Benchmark ({len(seeds)} seeds) ===")

    dataset = CharDataset(text=text, seq_len=seq_len)
    split = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    c_losses, c_ppls = [], []
    q_losses, q_ppls = [], []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Classical Transformer LM (~100K params)
        c_model = ClassicalTransformerLM(
            vocab_size=dataset.vocab_size,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            max_seq_len=seq_len,
        )
        c_val_loss, c_val_ppl, _ = train_lm(c_model, train_loader, val_loader, epochs=epochs)
        c_losses.append(c_val_loss)
        c_ppls.append(c_val_ppl)

        # Hybrid Quantum Transformer LM
        q_model = HybridQuantumTransformerLM(
            vocab_size=dataset.vocab_size,
            d_model=64,
            n_heads=2,
            n_layers=2,
            n_qubits=4,
            max_seq_len=seq_len,
        )
        q_val_loss, q_val_ppl, _ = train_lm(q_model, train_loader, val_loader, epochs=epochs)
        q_losses.append(q_val_loss)
        q_ppls.append(q_val_ppl)

        print(f"Seed {seed}: Classical Loss = {c_val_loss:.4f}, PPL = {c_val_ppl:.2f} | Hybrid Loss = {q_val_loss:.4f}, PPL = {q_val_ppl:.2f}")

    results = {
        "seeds": seeds,
        "vocab_size": dataset.vocab_size,
        "seq_len": seq_len,
        "classical": {
            "loss_mean": float(np.mean(c_losses)),
            "loss_std": float(np.std(c_losses)),
            "ppl_mean": float(np.mean(c_ppls)),
            "ppl_std": float(np.std(c_ppls)),
            "losses": c_losses,
            "ppls": c_ppls,
            "num_params": sum(p.numel() for p in c_model.parameters()),
        },
        "hybrid": {
            "loss_mean": float(np.mean(q_losses)),
            "loss_std": float(np.std(q_losses)),
            "ppl_mean": float(np.mean(q_ppls)),
            "ppl_std": float(np.std(q_ppls)),
            "losses": q_losses,
            "ppls": q_ppls,
            "num_params": sum(p.numel() for p in q_model.parameters()),
        },
    }

    print("\n================ LANGUAGE MODEL BENCHMARK SUMMARY ================")
    print(f"Classical LM Params:     {results['classical']['num_params']:,}")
    print(f"Classical LM Loss:       {results['classical']['loss_mean']:.4f} ± {results['classical']['loss_std']:.4f}")
    print(f"Classical LM Perplexity: {results['classical']['ppl_mean']:.2f} ± {results['classical']['ppl_std']:.2f}")
    print("------------------------------------------------------------------")
    print(f"Hybrid QML LM Params:    {results['hybrid']['num_params']:,}")
    print(f"Hybrid QML LM Loss:      {results['hybrid']['loss_mean']:.4f} ± {results['hybrid']['loss_std']:.4f}")
    print(f"Hybrid QML LM Perplexity:{results['hybrid']['ppl_mean']:.2f} ± {results['hybrid']['ppl_std']:.2f}")
    print("==================================================================")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved language model benchmark results to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classical LM vs Hybrid QML LM")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out", type=str, default="qml_lm_benchmark_results.json")
    args = parser.parse_args()

    run_lm_benchmark(
        seq_len=args.seq_len,
        epochs=args.epochs,
        seeds=args.seeds,
        output_json=args.out,
    )
