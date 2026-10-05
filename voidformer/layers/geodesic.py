"""Riemannian geodesic machinery.

Each token's path through the L layers,
    x(t=ℓ/L) := combined_ℓ ∈ ℝ^{d_model},
is treated as a curve on a statistical manifold equipped with a learnable
positive-definite Fisher-like metric tensor

    I_ℓ(x) = exp(s_ℓ(entropy_ℓ)) · (Uᵀ U + diag(softplus(d)) + ε I)

The geodesic arc-length is approximated by

    d(I, U) ≈ Σ_ℓ √( (x_{ℓ+1} - x_ℓ)ᵀ I_ℓ (x_{ℓ+1} - x_ℓ) )

We use a low-rank + diagonal parametrisation so the per-token Mahalanobis
norm costs O(k·d) instead of O(d²) (k = `rank`, default 16).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherInfoMetric(nn.Module):
    """Per-token positive-definite metric

        I(x, h) = κ(h) · ( Uᵀ U + diag(d) + ε I )

    where U ∈ ℝ^{k×D} is low-rank, d ∈ ℝ^{D} is diagonal (both produced from
    the local state), and κ is a positive conformal factor driven by entropy.
    """

    def __init__(self, d_model: int, rank: int = 16, eps: float = 1.0e-3) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.eps = eps
        self.proj_U = nn.Linear(d_model, rank * d_model, bias=False)
        self.proj_d = nn.Linear(d_model, d_model)
        self.kappa_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, entropy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:        (B, T, D)
        entropy:  (B, T)   per-token void-attention entropy

        returns:
            U      (B, T, k, D)         low-rank factor
            d_diag (B, T, D)            positive diagonal (after softplus + ε)
            kappa  (B, T)               positive conformal factor
        """
        B, T, D = x.shape
        U = self.proj_U(x).view(B, T, self.rank, D)
        d_diag = F.softplus(self.proj_d(x)) + self.eps
        kappa = torch.exp(self.kappa_net(entropy.unsqueeze(-1)).squeeze(-1))  # (B, T)
        return U, d_diag, kappa

    @staticmethod
    def quadratic(delta: torch.Tensor, U: torch.Tensor, d_diag: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Compute  Δᵀ I Δ  efficiently.

        delta:  (B, T, D)
        U:      (B, T, k, D)
        d_diag: (B, T, D)
        kappa:  (B, T)
        -> (B, T)
        """
        # ‖U Δ‖² = sum over rank of (U Δ)^2
        Ud = torch.einsum("btkd,btd->btk", U, delta)
        lowrank = (Ud * Ud).sum(dim=-1)            # (B, T)
        diag = (d_diag * delta * delta).sum(dim=-1)  # (B, T)
        return kappa * (lowrank + diag)


class RiemannianGeodesicTracker(nn.Module):
    """Computes the discrete geodesic length of each token's path through
    the network using a single shared FisherInfoMetric.

    Usage:
        tracker = RiemannianGeodesicTracker(d_model)
        tracker.reset(batch, seq, device)
        tracker.step(x_prev, x_curr, entropy_curr)       # call per layer
        d, steps = tracker.finalize()                    # (B,T), list[(B,T)]
    """

    def __init__(self, d_model: int, rank: int = 16) -> None:
        super().__init__()
        self.metric = FisherInfoMetric(d_model, rank=rank)
        self._segs: list[torch.Tensor] = []

    def reset(self) -> None:
        self._segs = []

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """Returns this segment's arc-length (B, T)."""
        delta = x_curr - x_prev
        # Metric evaluated at the segment midpoint -> better trapezoidal accuracy
        mid = 0.5 * (x_prev + x_curr)
        U, d_diag, kappa = self.metric(mid, entropy)
        sq = FisherInfoMetric.quadratic(delta, U, d_diag, kappa).clamp_min(0.0)
        seg = torch.sqrt(sq + 1.0e-8)                  # (B, T)
        self._segs.append(seg)
        return seg

    def finalize(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not self._segs:
            raise RuntimeError("RiemannianGeodesicTracker: no segments collected")
        total = torch.stack(self._segs, dim=0).sum(dim=0)   # (B, T)
        return total, list(self._segs)
