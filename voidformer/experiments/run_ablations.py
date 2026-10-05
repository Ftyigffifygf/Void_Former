"""Baseline Comparison and Ablation Suite for Quantum VoidFormer.

Compares:
1. Classical Transformer Baseline
2. Quantum VoidFormer (without Entanglement)
3. Quantum VoidFormer (with Hard Collapse STE)
4. Full Quantum VoidFormer (with Cross-Token Entanglement & Entropy-Gated Collapse)
"""

from __future__ import annotations

import sys
import types
import time
from pathlib import Path

# Add repo root and register namespace finder so imports work standalone
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

class VoidFormerFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "voidformer":
            from importlib.machinery import ModuleSpec
            from importlib.abc import Loader

            class VoidFormerLoader(Loader):
                def create_module(self, spec):
                    mod = types.ModuleType("voidformer")
                    mod.__path__ = [str(repo_root)]
                    return mod

                def exec_module(self, module):
                    pass

            return ModuleSpec("voidformer", VoidFormerLoader(), is_package=True)
        return None

sys.meta_path.insert(0, VoidFormerFinder())

import torch
import torch.nn as nn

from voidformer.models.quantum_voidformer import QuantumVoidFormer
from voidformer.models.voidformer import VoidFormerModel


def run_ablations():
    print("=" * 65)
    print("      QUANTUM VOIDFORMER ABLATION & BASELINE COMPARISON SUITE   ")
    print("=" * 65)

    vocab_size = 128
    d_model = 64
    seq_len = 16
    batch_size = 4
    steps = 10

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    criterion = nn.CrossEntropyLoss()

    # Classical model width scaled to ~1.0M parameters to match Quantum VoidFormer capacity
    models = {
        "1. Classical Baseline (Matched)": VoidFormerModel(
            vocab_size=vocab_size, d_model=96, d_void=96, d_ff=384, n_layers=2, n_heads=2
        ),
        "2. Quantum (No Entanglement)": QuantumVoidFormer(
            vocab_size=vocab_size, d_model=d_model, n_layers=2, n_heads=2, enable_entanglement=False
        ),
        "3. Quantum (Hard Collapse STE)": QuantumVoidFormer(
            vocab_size=vocab_size, d_model=d_model, n_layers=2, n_heads=2, collapse_protocol="hard"
        ),
        "4. Full Quantum VoidFormer": QuantumVoidFormer(
            vocab_size=vocab_size, d_model=d_model, n_layers=2, n_heads=2, collapse_protocol="entropy_gated", enable_entanglement=True
        ),
    }

    results = {}

    for name, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        start_time = time.time()
        loss_history = []

        for step in range(steps):
            optimizer.zero_grad()
            if isinstance(model, QuantumVoidFormer):
                out = model(x, return_diagnostics=True)
                logits = out.logits
            else:
                out = model(x)
                logits = out.logits

            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        elapsed = time.time() - start_time
        results[name] = {
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "time_sec": elapsed,
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        print(f"\nModel: {name}")
        print(f"  Parameters   : {results[name]['params']:,}")
        print(f"  Initial Loss : {results[name]['initial_loss']:.4f}")
        print(f"  Final Loss   : {results[name]['final_loss']:.4f}")
        print(f"  Elapsed Time : {results[name]['time_sec']:.2f} s")

    print("\n" + "=" * 65)
    print("                     SUMMARY TABLE                              ")
    print("=" * 65)
    print(f"{'Model Architecture':<32} | {'Params':<8} | {'Init Loss':<9} | {'Final Loss':<10}")
    print("-" * 65)
    for name, res in results.items():
        print(f"{name:<32} | {res['params']:<8} | {res['initial_loss']:<9.4f} | {res['final_loss']:<10.4f}")
    print("=" * 65)


if __name__ == "__main__":
    run_ablations()
