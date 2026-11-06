"""Unified training engine for classical, hybrid, and quantum models."""

from __future__ import annotations
import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from qai.training.loss import get_loss_function
from qai.training.optimizer import create_optimizer
from qai.training.metrics import compute_metrics
from qai.quantum.gradients import compute_gradient_norms
from qai.core.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """Unified trainer supporting CPU/GPU, quantum parameter logging, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task: str = "classification",
        lr: float = 0.01,
        quantum_lr: float = 0.01,
        classical_lr: float = 0.01,
        epochs: int = 5,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)

        self.model.to(self.device)
        self.criterion = get_loss_function(task)
        self.optimizer = create_optimizer(
            model=self.model,
            lr=lr,
            quantum_lr=quantum_lr,
            classical_lr=classical_lr,
        )

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "quantum_grad_norms": [],
            "classical_grad_norms": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        last_grad_norms = {}

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            if self.task == "sequence_modeling":
                # Flatten [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            elif self.task == "classification":
                loss = self.criterion(outputs, targets)
            else:  # regression
                loss = self.criterion(outputs.squeeze(), targets.squeeze())

            loss.backward()

            # Record gradient norms
            last_grad_norms = compute_gradient_norms(self.model)

            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        avg_loss = total_loss / max(total_samples, 1)
        self.history["train_loss"].append(avg_loss)
        self.history["quantum_grad_norms"].append(last_grad_norms.get("quantum_grad_norm", 0.0))
        self.history["classical_grad_norms"].append(last_grad_norms.get("classical_grad_norm", 0.0))

        return {
            "loss": avg_loss,
            "quantum_grad_norm": last_grad_norms.get("quantum_grad_norm", 0.0),
            "classical_grad_norm": last_grad_norms.get("classical_grad_norm", 0.0),
        }

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        data_loader = loader or self.val_loader
        if data_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                if self.task == "sequence_modeling":
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                elif self.task == "classification":
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs.squeeze(), targets.squeeze())

                total_loss += loss.item() * len(targets)
                total_samples += len(targets)
                all_preds.append(outputs)
                all_targets.append(targets)

        avg_loss = total_loss / max(total_samples, 1)
        full_preds = torch.cat(all_preds, dim=0)
        full_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(full_preds, full_targets, task=self.task)
        metrics["val_loss"] = avg_loss
        return metrics

    def train(self, save_best: bool = True, model_name: str = "model") -> Dict[str, Any]:
        best_val_loss = float("inf")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            train_res = self.train_epoch(epoch)
            val_res = self.evaluate() if self.val_loader is not None else {}

            val_loss = val_res.get("val_loss", train_res["loss"])
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")
                CheckpointManager.save(
                    filepath=ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                )

        total_time = time.time() - start_time
        return {
            "epochs": self.epochs,
            "total_training_time_sec": total_time,
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else 0.0,
            "best_val_loss": best_val_loss,
            "history": self.history,
        }
