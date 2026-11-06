"""Loss function factories for classification, regression, and sequence modeling."""

from __future__ import annotations
import torch
import torch.nn as nn


def get_loss_function(task: str = "classification") -> nn.Module:
    """Returns loss criterion appropriate for the task."""
    if task == "classification":
        return nn.CrossEntropyLoss()
    elif task == "sequence_modeling":
        return nn.CrossEntropyLoss()
    elif task == "regression":
        return nn.MSELoss()
    else:
        return nn.MSELoss()
