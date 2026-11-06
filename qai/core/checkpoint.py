"""Checkpoint management for saving and restoring model states and training metadata."""

from __future__ import annotations
import os
import torch
from typing import Any, Dict, Optional


class CheckpointManager:
    """Handles saving and restoring model and optimizer states."""

    @staticmethod
    def save(
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        loss: float = 0.0,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "extra_meta": extra_meta or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, filepath)
        return filepath

    @staticmethod
    def load(
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        payload = torch.load(filepath, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload
