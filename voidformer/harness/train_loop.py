"""Training Loop and Tracking for Harness Package."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    steps: int = 10,
    lr: float = 1e-3,
    log_file: Optional[str] = "harness_log.jsonl",
) -> dict[str, Any]:
    """Train model instance and log step metrics locally with optional JSONL backup.

    Args:
        model: PyTorch model
        dataloader: DataLoader instance
        steps: Total optimization steps
        lr: Learning rate
        log_file: Path to log file

    Returns:
        Dictionary of execution metrics
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss_history = []
    start_time = time.time()

    data_iter = iter(dataloader)

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x = batch["input_ids"].to(next(model.parameters()).device)
        y = batch["labels"].to(x.device)

        optimizer.zero_grad()
        output = model(x)
        logits = output.logits if hasattr(output, "logits") else output

        # Internal shifting convention: logits[:, :-1] vs y[:, 1:]
        if logits.ndim == 3 and y.ndim == 2 and logits.size(1) == y.size(1) and logits.size(1) > 1:
            logits = logits[:, :-1, :].contiguous()
            y = y[:, 1:].contiguous()

        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": loss_val, "timestamp": time.time()}) + "\n")

    elapsed = time.time() - start_time

    return {
        "initial_loss": loss_history[0] if loss_history else 0.0,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "elapsed_sec": elapsed,
        "loss_history": loss_history,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
