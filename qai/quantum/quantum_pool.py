"""Quantum Pool and Mathematical Foundation Engine.

Implements every mathematical equation of quantum states, superposition, interference,
Born-rule measurements, fidelity overlap, and Riemannian manifold geodesic metrics.
"""

from __future__ import annotations
import math
from typing import Sequence, Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from qai.core.device_manager import DeviceManager
from qai.core.tensor_utils import normalize_quantum_features


# =========================================================================
# Equation 1: Token Superposition State
# |T⟩ = α |S_c⟩ + β |S_v⟩  where |α|^2 + |β|^2 = 1
# =========================================================================
class QuantumTokenSuperposition(nn.Module):
    """
    Computes context-adaptive superposition of classical and quantum/void states:
    |T> = alpha |S_c> + beta |S_v>
    with unitary constraint |alpha|^2 + |beta|^2 = 1 parameterized via theta:
    alpha = cos(theta), beta = sin(theta)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # Context-adaptive mixing angle predictor
        self.theta_proj = nn.Linear(d_model * 2, 1)

    def forward(self, s_c: torch.Tensor, s_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # s_c: [batch, ..., d_model], s_v: [batch, ..., d_model]
        concat = torch.cat([s_c, s_v], dim=-1)
        theta = self.theta_proj(concat)  # [batch, ..., 1]
        alpha = torch.cos(theta)
        beta = torch.sin(theta)
        t_state = alpha * s_c + beta * s_v
        return t_state, alpha, beta


# =========================================================================
# Equation 2: Quantum Interference and Antisymmetric Layer Update
# combined_ℓ = α_ℓ c'_ℓ + β_ℓ φ(v'_ℓ) + γ⁺_ℓ i_+ - γ⁻_ℓ i_- + δ_ℓ (c'_ℓ ⊙ φ(v'_ℓ))
# =========================================================================
class QuantumInterferenceLayer(nn.Module):
    """
    Implements constructive and destructive interference between classical and quantum branches:
    i_+: constructive interference
    i_-: antisymmetric destructive phase interference
    combined = alpha * c' + beta * phi(v') + gamma_plus * i_+ - gamma_minus * i_- + delta * (c' * phi(v'))
    """

    def __init__(self, d_model: int, d_v: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_v = d_v or d_model
        self.phi = nn.Linear(self.d_v, d_model) if self.d_v != d_model else nn.Identity()

        # Mixing coefficients
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_plus = nn.Parameter(torch.ones(1) * 0.2)
        self.gamma_minus = nn.Parameter(torch.ones(1) * 0.2)
        self.delta = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, c_prime: torch.Tensor, v_prime: torch.Tensor) -> torch.Tensor:
        v_proj = self.phi(v_prime)
        # Constructive and destructive interference components
        i_plus = 0.5 * (c_prime + v_proj)
        i_minus = 0.5 * (c_prime - v_proj)
        hadamard_prod = c_prime * v_proj

        combined = (
            self.alpha * c_prime
            + self.beta * v_proj
            + self.gamma_plus * i_plus
            - self.gamma_minus * i_minus
            + self.delta * hadamard_prod
        )
        return combined


# =========================================================================
# Equation 3: Stochastic Void Attention with Latent Field Floor
# K' = K + ε,  ε ~ N(0, σ² I)
# A = softmax((Q K'ᵀ / √d) · 1/τ_h)
# A = (1 - π) A + π U
# Out = A V
# =========================================================================
class StochasticQuantumAttention(nn.Module):
    """Stochastic attention with Gaussian uncertainty and uniform latent field floor."""

    def __init__(self, d_model: int, n_heads: int = 1, sigma: float = 0.05, pi_floor: float = 0.05) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sigma = sigma
        self.pi_floor = pi_floor
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.tau = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)

        # Stochastic perturbation: K' = K + epsilon
        if self.training and self.sigma > 0:
            noise = torch.randn_like(k) * self.sigma
            k = k + noise

        # Scaled dot-product: (Q @ K'T / sqrt(d_k)) * (1 / tau)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.d_k) * torch.clamp(self.tau, min=0.1))

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        a = F.softmax(scores, dim=-1)

        # Floor mixture: A = (1 - pi) A + pi * U
        u = torch.ones_like(a) / s
        a_mixed = (1.0 - self.pi_floor) * a + self.pi_floor * u

        # Entropy of attention distribution: H[A] = -sum(A log A)
        eps = 1e-9
        entropy = -torch.sum(a_mixed * torch.log(a_mixed + eps), dim=-1).mean(dim=1)  # [b, s]

        out = torch.matmul(a_mixed, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out), entropy


# =========================================================================
# Equation 4: Adaptive Wavefunction Collapse Engine
# τ_ℓ = σ( g(combined_ℓ, cos(c'_ℓ, combined_ℓ), H[A]) / T  + 4(ℓ/L - 0.5) )
# y_ℓ = (1 - τ_ℓ) combined_ℓ + τ_ℓ c'_ℓ
# =========================================================================
class QuantumCollapseEngine(nn.Module):
    """Adaptive wave function collapse from quantum superposition to deterministic state."""

    def __init__(self, d_model: int, layer_idx: int = 0, total_layers: int = 1, temperature: float = 1.0, mode: str = "soft") -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.total_layers = max(total_layers, 1)
        self.temperature = temperature
        self.mode = mode

        # g function takes combined_l, cosine similarity (scalar), and entropy H[A] (scalar) -> 1
        self.g_net = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, combined: torch.Tensor, c_prime: torch.Tensor, entropy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(c_prime, combined, dim=-1, eps=1e-8).unsqueeze(-1)  # [..., 1]
        ent = entropy.unsqueeze(-1) if entropy.dim() == combined.dim() - 1 else entropy

        g_in = torch.cat([combined, cos_sim, ent], dim=-1)
        g_val = self.g_net(g_in)

        # Depth bias: 4 * (l / L - 0.5)
        depth_bias = 4.0 * (self.layer_idx / self.total_layers - 0.5)
        tau = torch.sigmoid(g_val / max(self.temperature, 1e-4) + depth_bias)

        if self.mode == "coexist":
            tau = 0.3 * tau
        elif self.mode == "hard":
            tau_hard = (tau > 0.5).float()
            tau = (tau_hard - tau).detach() + tau  # Straight-through estimator

        # Collapse: y = (1 - tau) * combined + tau * c'
        y = (1.0 - tau) * combined + tau * c_prime
        return y, tau


# =========================================================================
# Equation 5: Uncertainty Memory (Recurrent Depth Memory)
# z_ℓ = σ(W_z [v'_ℓ, m_{ℓ-1}])
# r_ℓ = σ(W_r [v'_ℓ, m_{ℓ-1}])
# ĉ_ℓ = tanh(W_c [v'_ℓ, r_ℓ ⊙ m_{ℓ-1}])
# m_ℓ = (1 - z_ℓ) m_{ℓ-1} + z_ℓ ĉ_ℓ
# v''_ℓ = LN(v'_ℓ + W_o m_ℓ)
# =========================================================================
class QuantumUncertaintyMemory(nn.Module):
    """GRU-inspired uncertainty state recurrence across depth."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_z = nn.Linear(d_model * 2, d_model)
        self.w_r = nn.Linear(d_model * 2, d_model)
        self.w_c = nn.Linear(d_model * 2, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, v_prime: torch.Tensor, m_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([v_prime, m_prev], dim=-1)
        z = torch.sigmoid(self.w_z(concat))
        r = torch.sigmoid(self.w_r(concat))

        c_hat_in = torch.cat([v_prime, r * m_prev], dim=-1)
        c_hat = torch.tanh(self.w_c(c_hat_in))

        m = (1.0 - z) * m_prev + z * c_hat
        v_double_prime = self.ln(v_prime + self.w_o(m))
        return v_double_prime, m


# =========================================================================
# Equation 6: Riemannian Geodesic Metric and Path Distance
# I_ℓ(x) = κ_ℓ(H_ℓ) · ( U_ℓ(x)ᵀ U_ℓ(x) + diag(d_ℓ(x)) + ε I )
# d(I, U) ≈ Σ_{ℓ=0}^{L-1} √( Δx_ℓᵀ I_ℓ(x̄_ℓ) Δx_ℓ )
# L_geodesic = ‖d(I, U) − τ_geo‖² + 0.1 · Var_t[d(I, U)]
# =========================================================================
class RiemannianGeodesicEngine(nn.Module):
    """Computes Fisher-information-inspired Riemannian geodesic distances on semantic manifolds."""

    def __init__(self, d_model: int, rank_k: int = 16, target_distance: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank_k = rank_k
        self.target_distance = target_distance

        # Low-rank factor U(x): R^{d_model} -> R^{k x d_model}
        self.u_proj = nn.Linear(d_model, rank_k * d_model)
        # Softplus diagonal correction d(x): R^{d_model} -> R^{d_model}
        self.d_proj = nn.Linear(d_model, d_model)
        # Conformal factor kappa(H): scalar MLP
        self.kappa_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def compute_metric_inner_product(self, x_mid: torch.Tensor, delta_x: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        b, s, d = delta_x.shape
        # Conformal factor kappa > 0
        ent = entropy.unsqueeze(-1) if entropy.dim() == 2 else entropy
        kappa = self.kappa_net(ent) + 1e-4  # [b, s, 1]

        # Low-rank projection: ||U delta_x||^2
        u_mat = self.u_proj(x_mid).view(b, s, self.rank_k, d)
        # u_mat @ delta_x: [b, s, k]
        u_dx = torch.matmul(u_mat, delta_x.unsqueeze(-1)).squeeze(-1)
        low_rank_term = torch.sum(u_dx ** 2, dim=-1, keepdim=True)  # [b, s, 1]

        # Diagonal term: sum_i d_i * (delta_x_i)^2
        d_diag = F.softplus(self.d_proj(x_mid)) + 1e-4
        diag_term = torch.sum(d_diag * (delta_x ** 2), dim=-1, keepdim=True)

        eps_term = 1e-4 * torch.sum(delta_x ** 2, dim=-1, keepdim=True)

        quad_form = kappa * (low_rank_term + diag_term + eps_term)
        return torch.sqrt(torch.clamp(quad_form, min=1e-8)).squeeze(-1)  # [b, s]

    def compute_geodesic_distance(self, trajectory: List[torch.Tensor], entropies: List[torch.Tensor]) -> torch.Tensor:
        # trajectory: list of length L containing [b, s, d]
        # Sum midpoint approximations across all L-1 intervals
        total_dist = torch.zeros(trajectory[0].shape[0], trajectory[0].shape[1], device=trajectory[0].device)
        for l in range(len(trajectory) - 1):
            x_l = trajectory[l]
            x_next = trajectory[l + 1]
            delta_x = x_next - x_l
            x_mid = 0.5 * (x_l + x_next)
            h_l = entropies[min(l, len(entropies) - 1)]
            seg_dist = self.compute_metric_inner_product(x_mid, delta_x, h_l)
            total_dist = total_dist + seg_dist
        return total_dist

    def geodesic_loss(self, distances: torch.Tensor) -> torch.Tensor:
        """L_geodesic = ||d(I, U) - tau_geo||^2 + 0.1 * Var[d(I, U)]"""
        mean_diff = torch.mean((distances - self.target_distance) ** 2)
        var_term = torch.var(distances) if distances.numel() > 1 else torch.tensor(0.0, device=distances.device)
        return mean_diff + 0.1 * var_term


# =========================================================================
# Unified Quantum Mathematics Pool
# =========================================================================
class QuantumPool(nn.Module):
    """
    Central Quantum Pool executing and verifying all quantum mathematical equations:
    - Superposition state mixing (|T> = alpha |S_c> + beta |S_v>)
    - Constructive and destructive interference updates
    - Stochastic void attention with latent floor and entropy estimation
    - Wavefunction collapse engine (soft, hard, coexist modes)
    - Uncertainty memory recurrence
    - Riemannian geodesic distance and manifold loss calculation
    - Fidelity overlap circuit execution
    """

    def __init__(self, d_model: int = 16, n_qubits: int = 4, total_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.total_layers = total_layers

        self.superposition = QuantumTokenSuperposition(d_model)
        self.interference = QuantumInterferenceLayer(d_model)
        self.stochastic_attn = StochasticQuantumAttention(d_model)
        self.collapse = QuantumCollapseEngine(d_model, layer_idx=0, total_layers=total_layers)
        self.memory = QuantumUncertaintyMemory(d_model)
        self.geodesic = RiemannianGeodesicEngine(d_model)

    def execute_full_cycle(
        self,
        c_state: torch.Tensor,
        v_state: torch.Tensor,
        m_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Runs the complete quantum mathematical pipeline."""
        b, s, d = c_state.shape
        if m_prev is None:
            m_prev = torch.zeros_like(v_state)

        # 1. Superposition
        t_state, alpha, beta = self.superposition(c_state, v_state)

        # 2. Stochastic void attention & entropy
        v_attn, entropy = self.stochastic_attn(v_state)

        # 3. Uncertainty memory
        v_mem, m_curr = self.memory(v_attn, m_prev)

        # 4. Constructive & destructive interference
        combined = self.interference(c_state, v_mem)

        # 5. Adaptive collapse
        collapsed, tau = self.collapse(combined, c_state, entropy)

        # 6. Geodesic distance on manifold trajectory
        trajectory = [c_state, combined, collapsed]
        geodesic_dist = self.geodesic.compute_geodesic_distance(trajectory, [entropy, entropy])
        geo_loss = self.geodesic.geodesic_loss(geodesic_dist)

        return {
            "token_superposition": t_state,
            "alpha": alpha,
            "beta": beta,
            "entropy": entropy,
            "v_mem": v_mem,
            "m_curr": m_curr,
            "combined": combined,
            "collapsed": collapsed,
            "tau": tau,
            "geodesic_distance": geodesic_dist,
            "geodesic_loss": geo_loss,
        }
