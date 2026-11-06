"""Command Line Interface for Q-INTELLIGENCE."""

from __future__ import annotations
import argparse
import sys
import logging
import torch

from qai.core.configuration import QAIConfig
from qai.core.device_manager import DeviceManager
from qai.core.backend_registry import BackendRegistry
from qai.hardware.hardware_validation import HardwareValidator
from qai.datasets import (
    get_xor_dataset,
    get_deterministic_sequence_dataset,
    create_train_val_loaders,
)
from qai.neural import ClassicalANN, QuantumANN, HybridANN
from qai.transformer import (
    ClassicalTransformer,
    HybridQuantumTransformer,
    QuantumNativeTransformer,
)
from qai.training.trainer import Trainer
from qai.evaluation.benchmark import BenchmarkSuite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qai.cli")


def cmd_doctor(args: argparse.Namespace) -> int:
    """Runs system diagnostics, environment verification, and device availability checks."""
    print("=" * 60)
    print(" Q-INTELLIGENCE SYSTEM DIAGNOSTICS & DOCTOR")
    print("=" * 60)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")

    import pennylane as qml
    import qiskit
    print(f"PennyLane version: {qml.__version__}")
    print(f"Qiskit version: {qiskit.__version__}")

    print("\nSupported Registered Backends:")
    for b in BackendRegistry.list_backends():
        print(f" - {b['name']}: {b['description']} (Hardware: {b['is_hardware']})")

    print("\nTesting backend instantiation:")
    for b_name in ["default.qubit", "lightning.qubit", "default.mixed"]:
        try:
            dev = DeviceManager.create_device(b_name, n_qubits=2)
            print(f" [PASS] Device: {b_name} -> {dev.name}")
        except Exception as e:
            print(f" [WARN] Device {b_name} unavailable: {e}")

    print("\nHardware Validation Status:")
    hw_status = HardwareValidator.check_hardware_status()
    for k, v in hw_status.items():
        print(f" {k}: {v}")

    print("=" * 60)
    return 0


def cmd_hardware_check(args: argparse.Namespace) -> int:
    """Checks IBM Quantum credentials and hardware connectivity."""
    print("Checking IBM Quantum hardware access...")
    status = HardwareValidator.check_hardware_status(allow_hardware=args.allow_hardware)
    print("Result:")
    for k, v in status.items():
        print(f"  {k}: {v}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Trains specified model architecture."""
    config = QAIConfig.from_yaml(args.config) if args.config else QAIConfig()

    model_name = args.model or config.model.model_type
    backend_name = args.backend or config.quantum.backend
    epochs = args.epochs or config.training.epochs
    lr = args.lr or config.training.learning_rate

    print(f"Starting training: model={model_name}, backend={backend_name}, epochs={epochs}")

    if args.hardware:
        status = HardwareValidator.check_hardware_status(backend_name=config.hardware.ibm_backend, allow_hardware=True)
        if status["hardware_access"] != "PASS":
            print(f"ABORTING: Hardware training requested but not available. Status: {status}")
            return 1

    # Instantiate model & dataset
    if model_name in ["classical_ann", "quantum_ann", "hybrid_ann"]:
        ds = get_xor_dataset(n_samples=40, seed=config.training.seed)
        train_l, val_l = create_train_val_loaders(ds, batch_size=config.training.batch_size)
        task = "classification"

        if model_name == "classical_ann":
            model = ClassicalANN(in_features=2, hidden_features=8, out_features=2)
        elif model_name == "quantum_ann":
            model = QuantumANN(in_features=2, out_features=2, n_qubits=config.quantum.n_qubits, n_layers=config.quantum.n_layers, backend_name=backend_name)
        else:
            model = HybridANN(in_features=2, hidden_features=8, out_features=2, n_qubits=config.quantum.n_qubits, backend_name=backend_name)

    elif model_name in ["classical_transformer", "hybrid_transformer", "quantum_native_transformer"]:
        ds = get_deterministic_sequence_dataset(n_samples=20, seq_len=4, vocab_size=8, seed=config.training.seed)
        train_l, val_l = create_train_val_loaders(ds, batch_size=config.training.batch_size)
        task = "sequence_modeling"

        if model_name == "classical_transformer":
            model = ClassicalTransformer(vocab_size=8, d_model=16, n_heads=1, max_seq_len=8)
        elif model_name == "hybrid_transformer":
            model = HybridQuantumTransformer(vocab_size=8, d_model=16, n_qubits=config.quantum.n_qubits, backend_name=backend_name)
        else:
            model = QuantumNativeTransformer(vocab_size=8, n_qubits=3, n_layers=config.quantum.n_layers, backend_name=backend_name)
    else:
        print(f"Unknown model: {model_name}")
        return 1

    trainer = Trainer(model, train_l, val_l, task=task, epochs=epochs, lr=lr)
    results = trainer.train(model_name=model_name)
    print(f"Training completed successfully in {results['total_training_time_sec']:.2f}s. Final loss: {results['final_train_loss']:.4f}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Executes full benchmark suite across all models and saves results."""
    print("Executing full benchmark suite...")
    res = BenchmarkSuite.run_all_benchmarks(epochs=args.epochs, output_dir=args.output_dir)
    print("Benchmark complete! Saved outputs to:", args.output_dir)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-INTELLIGENCE Quantum Artificial Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Doctor
    subparsers.add_parser("doctor", help="Run system diagnostics and verify dependencies")

    # Hardware check
    hw_p = subparsers.add_parser("hardware-check", help="Verify IBM Quantum credentials and hardware connectivity")
    hw_p.add_argument("--allow-hardware", action="store_true", help="Authorize hardware validation query")

    # Train
    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("--model", type=str, default=None, help="Model to train")
    train_p.add_argument("--backend", type=str, default=None, help="PennyLane execution backend")
    train_p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    train_p.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_p.add_argument("--config", type=str, default=None, help="Path to config YAML")
    train_p.add_argument("--hardware", action="store_true", help="Execute on physical hardware")

    # Benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run scientific benchmarks")
    bench_p.add_argument("--all", action="store_true", help="Run benchmarks across all models")
    bench_p.add_argument("--epochs", type=int, default=2, help="Epochs per model")
    bench_p.add_argument("--output-dir", type=str, default="logs/benchmarks", help="Output directory")

    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.command == "hardware-check":
        sys.exit(cmd_hardware_check(args))
    elif args.command == "train":
        sys.exit(cmd_train(args))
    elif args.command == "benchmark":
        sys.exit(cmd_benchmark(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
