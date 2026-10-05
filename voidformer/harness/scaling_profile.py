"""Scaling and Memory Profiler for Harness Package."""

from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from voidformer.harness.model_factory import create_model


def profile_scaling() -> list[dict]:
    """Profile latency and memory vs qubit count and sequence length."""
    qubit_counts = [2, 3, 4]
    seq_lengths = [16, 32]

    results = []
    print("=" * 65)
    print("              HARNESS SCALING & MEMORY PROFILER                  ")
    print("=" * 65)

    for n_q in qubit_counts:
        for seq_len in seq_lengths:
            model = create_model("quantum", n_qubits_per_token=n_q, max_seq_len=seq_len)
            x = torch.randint(0, 128, (2, seq_len))

            start = time.time()
            out = model(x)
            loss = out.logits.sum()
            loss.backward()
            elapsed = time.time() - start

            res = {
                "n_qubits": n_q,
                "seq_len": seq_len,
                "latency_sec": elapsed,
            }
            results.append(res)
            print(f"Qubits: {n_q} | SeqLen: {seq_len:<4} | Latency: {elapsed:.4f}s")

    return results


if __name__ == "__main__":
    profile_scaling()
