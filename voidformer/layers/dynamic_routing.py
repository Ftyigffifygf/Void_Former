"""Dynamic Routing — chooses among classical / void / merged paths per token.

Uses Gumbel-softmax over 3 routes; produces a soft mixture during training
and a near one-hot routing at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRouter(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.router = nn.Linear(d_model, 3)

    def forward(
        self,
        classical: torch.Tensor,
        void_in_c_space: torch.Tensor,
        merged: torch.Tensor,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(merged)                       # (B, T, 3)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=tau, hard=False)
        else:
            weights = F.softmax(logits, dim=-1)
        wc, wv, wm = weights.unbind(dim=-1)
        out = (
            wc.unsqueeze(-1) * classical
            + wv.unsqueeze(-1) * void_in_c_space
            + wm.unsqueeze(-1) * merged
        )
        return out, weights
