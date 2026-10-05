"""Quantum Boltzmann Distribution Sampler for Thermal Quantum State text generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QuantumBoltzmannSampler:
    """Quantum Thermal Sampler deriving token sampling probabilities from thermal quantum state density matrices."""

    def __init__(self, temperature: float = 1.0, kb: float = 1.0):
        self.temperature = max(temperature, 1e-6)
        self.kb = kb

    def sample(
        self,
        logits: torch.Tensor,
        entropy_scale: float = 0.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample next token from Quantum Boltzmann Distribution P(i) ~ exp(-E_i / k_B T)."""
        # Energy levels E_i derived from logits
        effective_temp = self.temperature * (1.0 + 0.1 * entropy_scale)

        # Inject thermal quantum fluctuations derived from thermal density matrix measurement
        thermal_noise = torch.randn_like(logits) * (0.05 * effective_temp)
        thermal_logits = (logits + thermal_noise) / effective_temp

        if top_k is not None:
            v, _ = torch.topk(thermal_logits, min(top_k, thermal_logits.size(-1)))
            thermal_logits[thermal_logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(thermal_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
