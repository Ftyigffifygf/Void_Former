"""A/B Benchmark Module for Harness Package."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from voidformer.harness.model_factory import create_model
from voidformer.harness.data import create_dataloader
from voidformer.harness.train_loop import train_model


def run_ab_benchmark(steps: int = 10, batch_size: int = 4, seq_len: int = 32) -> dict:
    """Run parameter-matched side-by-side A/B benchmark between Classical and Quantum VoidFormer."""
    dataloader = create_dataloader(vocab_size=128, seq_len=seq_len, batch_size=batch_size)

    print("=" * 65)
    print("              HARNESS A/B BENCHMARK RUNNER                      ")
    print("=" * 65)

    # Parameter-matched Classical Baseline vs Quantum VoidFormer
    classical = create_model("classical", d_model=96, d_void=96, d_ff=384, seq_len=seq_len)
    quantum = create_model("quantum", d_model=64, seq_len=seq_len)

    print(f"Classical Baseline Params : {sum(p.numel() for p in classical.parameters()):,}")
    print(f"Quantum VoidFormer Params : {sum(p.numel() for p in quantum.parameters()):,}")

    res_c = train_model(classical, dataloader, steps=steps, log_file="harness_classical.jsonl")
    res_q = train_model(quantum, dataloader, steps=steps, log_file="harness_quantum.jsonl")

    print(f"\nClassical Final Loss : {res_c['final_loss']:.4f} ({res_c['elapsed_sec']:.2f}s)")
    print(f"Quantum Final Loss   : {res_q['final_loss']:.4f} ({res_q['elapsed_sec']:.2f}s)")

    return {"classical": res_c, "quantum": res_q}


if __name__ == "__main__":
    run_ab_benchmark()
