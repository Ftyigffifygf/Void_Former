"""Core configuration management for Q-INTELLIGENCE."""

from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QuantumConfig:
    backend: str = "default.qubit"
    n_qubits: int = 4
    n_layers: int = 2
    encoding: str = "angle"
    entanglement: str = "ring"
    diff_method: str = "backprop"
    shots: Optional[int] = None
    noise_probability: float = 0.0
    ansatz: str = "hardware_efficient"


@dataclass
class ModelConfig:
    model_type: str = "hybrid_transformer"
    vocab_size: int = 32
    d_model: int = 16
    n_heads: int = 1
    d_ff: int = 32
    max_seq_len: int = 16
    dropout: float = 0.0
    num_layers: int = 1
    task: str = "sequence_modeling"  # binary_classification, multiclass_classification, regression, sequence_modeling
    num_classes: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 0.01
    quantum_lr: float = 0.01
    classical_lr: float = 0.01
    epochs: int = 5
    optimizer: str = "adam"
    weight_decay: float = 0.0
    seed: int = 42
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cpu"


@dataclass
class HardwareConfig:
    ibm_token: Optional[str] = None
    ibm_instance: Optional[str] = None
    ibm_backend: str = "ibm_sherbrooke"
    shots: int = 1024
    max_circuits_per_job: int = 100
    allow_hardware_execution: bool = False
    transpile_optimization_level: int = 1


@dataclass
class QAIConfig:
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAIConfig:
        quantum = QuantumConfig(**data.get("quantum", {}))
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))
        hardware = HardwareConfig(**data.get("hardware", {}))
        return cls(quantum=quantum, model=model, training=training, hardware=hardware)

    @classmethod
    def from_yaml(cls, path: str) -> QAIConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantum": self.quantum.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "hardware": self.hardware.__dict__,
        }

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
