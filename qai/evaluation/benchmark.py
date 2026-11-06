"""Comprehensive scientific benchmarking suite for all models."""

from __future__ import annotations
import time
import torch
from typing import Dict, Any, List

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
from qai.evaluation.reports import BenchmarkReporter
from qai.evaluation.resource_analysis import analyze_trainability_and_barren_plateaus


class BenchmarkSuite:
    """Runs scientific evaluation and comparison across all model architectures."""

    @staticmethod
    def run_all_benchmarks(
        epochs: int = 2,
        output_dir: str = "logs/benchmarks",
    ) -> Dict[str, Any]:
        results = []

        # ----------------------------------------------------
        # 1. Classification benchmark (XOR)
        # ----------------------------------------------------
        ds_xor = get_xor_dataset(n_samples=40, seed=42)
        train_loader_c, val_loader_c = create_train_val_loaders(ds_xor, batch_size=4)

        # Classical ANN
        c_ann = ClassicalANN(in_features=2, hidden_features=8, out_features=2)
        t_c_ann = Trainer(c_ann, train_loader_c, val_loader_c, epochs=epochs, lr=0.05)
        res_c_ann = t_c_ann.train(save_best=False)
        eval_c_ann = t_c_ann.evaluate()

        results.append({
            "model": "Classical ANN",
            "type": "pure_classical",
            "task": "binary_classification",
            "epochs": epochs,
            "train_loss": res_c_ann["final_train_loss"],
            "val_loss": eval_c_ann.get("val_loss", 0.0),
            "accuracy": eval_c_ann.get("accuracy", 0.0),
            "training_time_sec": res_c_ann["total_training_time_sec"],
        })

        # Quantum ANN
        q_ann = QuantumANN(in_features=2, out_features=2, n_qubits=2, n_layers=1)
        t_q_ann = Trainer(q_ann, train_loader_c, val_loader_c, epochs=epochs, lr=0.05)
        res_q_ann = t_q_ann.train(save_best=False)
        eval_q_ann = t_q_ann.evaluate()

        results.append({
            "model": "Quantum ANN",
            "type": "pennylane_qnn",
            "task": "binary_classification",
            "epochs": epochs,
            "train_loss": res_q_ann["final_train_loss"],
            "val_loss": eval_q_ann.get("val_loss", 0.0),
            "accuracy": eval_q_ann.get("accuracy", 0.0),
            "training_time_sec": res_q_ann["total_training_time_sec"],
        })

        # Hybrid ANN
        h_ann = HybridANN(in_features=2, hidden_features=4, out_features=2, n_qubits=2, n_layers=1)
        t_h_ann = Trainer(h_ann, train_loader_c, val_loader_c, epochs=epochs, lr=0.05)
        res_h_ann = t_h_ann.train(save_best=False)
        eval_h_ann = t_h_ann.evaluate()

        results.append({
            "model": "Hybrid ANN",
            "type": "hybrid",
            "task": "binary_classification",
            "epochs": epochs,
            "train_loss": res_h_ann["final_train_loss"],
            "val_loss": eval_h_ann.get("val_loss", 0.0),
            "accuracy": eval_h_ann.get("accuracy", 0.0),
            "training_time_sec": res_h_ann["total_training_time_sec"],
        })

        # ----------------------------------------------------
        # 2. Sequence Modeling Benchmark (Transformers)
        # ----------------------------------------------------
        ds_seq = get_deterministic_sequence_dataset(n_samples=20, seq_len=4, vocab_size=8, seed=42)
        train_loader_s, val_loader_s = create_train_val_loaders(ds_seq, batch_size=2)

        # Model A: Classical Transformer
        m_a = ClassicalTransformer(vocab_size=8, d_model=8, n_heads=1, max_seq_len=8)
        t_m_a = Trainer(m_a, train_loader_s, val_loader_s, task="sequence_modeling", epochs=epochs, lr=0.02)
        res_m_a = t_m_a.train(save_best=False)
        eval_m_a = t_m_a.evaluate()

        results.append({
            "model": "Classical Transformer",
            "type": "pure_classical",
            "task": "sequence_modeling",
            "epochs": epochs,
            "train_loss": res_m_a["final_train_loss"],
            "val_loss": eval_m_a.get("val_loss", 0.0),
            "accuracy": eval_m_a.get("token_accuracy", 0.0),
            "training_time_sec": res_m_a["total_training_time_sec"],
        })

        # Model B: Hybrid Quantum Transformer
        m_b = HybridQuantumTransformer(vocab_size=8, d_model=8, n_qubits=2, max_seq_len=8)
        t_m_b = Trainer(m_b, train_loader_s, val_loader_s, task="sequence_modeling", epochs=epochs, lr=0.02)
        res_m_b = t_m_b.train(save_best=False)
        eval_m_b = t_m_b.evaluate()

        results.append({
            "model": "Hybrid Quantum Transformer",
            "type": "hybrid_quantum",
            "task": "sequence_modeling",
            "epochs": epochs,
            "train_loss": res_m_b["final_train_loss"],
            "val_loss": eval_m_b.get("val_loss", 0.0),
            "accuracy": eval_m_b.get("token_accuracy", 0.0),
            "training_time_sec": res_m_b["total_training_time_sec"],
        })

        # Model C: Quantum-Native Transformer Prototype
        m_c = QuantumNativeTransformer(vocab_size=8, n_qubits=3, n_layers=1, max_seq_len=8)
        t_m_c = Trainer(m_c, train_loader_s, val_loader_s, task="sequence_modeling", epochs=epochs, lr=0.02)
        res_m_c = t_m_c.train(save_best=False)
        eval_m_c = t_m_c.evaluate()

        results.append({
            "model": "Quantum-Native Prototype",
            "type": "variational_quantum",
            "task": "sequence_modeling",
            "epochs": epochs,
            "train_loss": res_m_c["final_train_loss"],
            "val_loss": eval_m_c.get("val_loss", 0.0),
            "accuracy": eval_m_c.get("token_accuracy", 0.0),
            "training_time_sec": res_m_c["total_training_time_sec"],
        })

        # ----------------------------------------------------
        # 3. Barren Plateau / Trainability Benchmark
        # ----------------------------------------------------
        trainability_results = analyze_trainability_and_barren_plateaus(
            qubit_range=[2, 4],
            depth_range=[1, 2],
            n_samples=5,
        )

        # Export files
        BenchmarkReporter.export_json(results, f"{output_dir}/benchmark_results.json")
        BenchmarkReporter.export_csv(results, f"{output_dir}/benchmark_results.csv")
        BenchmarkReporter.export_json(trainability_results, f"{output_dir}/trainability_analysis.json")
        BenchmarkReporter.export_csv(trainability_results, f"{output_dir}/trainability_analysis.csv")

        return {
            "model_benchmarks": results,
            "trainability_analysis": trainability_results,
        }
