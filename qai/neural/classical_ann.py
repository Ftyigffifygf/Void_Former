"""Classical Artificial Neural Network baseline for rigorous comparison."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence, Optional


class ClassicalANN(nn.Module):
    """Parameter-matched classical feed-forward neural network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 16,
        out_features: int = 2,
        num_layers: int = 2,
        task: str = "classification",  # 'classification' or 'regression'
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task = task

        layers = []
        curr_in = in_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_in, hidden_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_in = hidden_features
        layers.append(nn.Linear(curr_in, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.task == "regression" and self.out_features == 1:
            return out.squeeze(-1)
        return out
