"""Ablation Grid Sweep Module for Harness Package."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from voidformer.harness.model_factory import create_model
from voidformer.harness.data import create_dataloader
from voidformer.harness.train_loop import train_model


def run_ablation_sweep(steps: int = 5) -> list[dict]:
    """Run grid sweep over collapse_protocol, enable_entanglement, and n_qubits_per_token."""
    dataloader = create_dataloader(vocab_size=128, seq_len=16, batch_size=4)

    protocols = ["hard", "soft", "entropy_gated"]
    entanglements = [True, False]

    results = []
    print("=" * 65)
    print("            HARNESS ABLATION GRID SWEEP RUNNER                  ")
    print("=" * 65)

    for proto in protocols:
        for ent in entanglements:
            name = f"proto={proto}_ent={ent}"
            model = create_model("quantum", collapse_protocol=proto, enable_entanglement=ent)
            res = train_model(model, dataloader, steps=steps, log_file=f"harness_sweep_{name}.jsonl")
            res["name"] = name
            results.append(res)
            print(f"Config: {name:<30} | Final Loss: {res['final_loss']:.4f}")

    return results


if __name__ == "__main__":
    run_ablation_sweep()
