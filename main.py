"""Main CLI Dispatcher for VoidFormer Suite."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="VoidFormer Universal Dispatcher CLI")
    parser.add_argument("mode", choices=["train", "infer", "demo", "test", "harness"], help="Execution command")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments for underlying command")
    args = parser.parse_args()

    if args.mode == "train":
        from train import main as train_main
        sys.argv = [sys.argv[0]] + args.extra
        train_main()
    elif args.mode == "infer":
        from infer import main as infer_main
        sys.argv = [sys.argv[0]] + args.extra
        infer_main()
    elif args.mode == "demo":
        from voidformer.quantum_init import initialize_quantum_processor
        import torch
        proc = initialize_quantum_processor(d_model=64, n_qubits_per_token=4)
        x = torch.randn(2, 8, 64)
        out, diag = proc(x)
        print("VoidFormer Quantum Demo Execution Complete!")
        print(f"Output shape: {out.shape}")
        print(f"Initial entropy: {diag.get('initial_quantum_entropy', 0.0):.4f}")
    elif args.mode == "test":
        import pytest
        sys.exit(pytest.main(args.extra))
    elif args.mode == "harness":
        from voidformer.harness.cli import main as harness_main
        sys.argv = [sys.argv[0]] + args.extra
        harness_main()


if __name__ == "__main__":
    main()
