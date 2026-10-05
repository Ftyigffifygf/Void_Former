"""Checkpoint loading and saving utilities."""

from __future__ import annotations

import torch

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    config: dict = None,
    step: int = 0,
    seed: int = 42,
    *args,
    **kwargs,
) -> None:
    state = {
        "model": model.state_dict(),
        "step": step,
        "seed": seed,
    }
    if optimizer:
        state["optimizer"] = optimizer.state_dict()
    if config:
        state["config"] = config
    torch.save(state, path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    *args,
    **kwargs,
) -> tuple[int, int, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    step = checkpoint.get("step", 0) if isinstance(checkpoint, dict) else 0
    seed = checkpoint.get("seed", 42) if isinstance(checkpoint, dict) else 42
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return step, seed, config
