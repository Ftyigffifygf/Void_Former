"""Autoregressive generation utilities for transformer sequence models."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Generates new tokens autoregressively.
    prompt_ids: shape (batch_size, seq_len)
    """
    model.eval()
    curr_ids = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    return curr_ids
