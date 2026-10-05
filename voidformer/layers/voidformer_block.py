"""VoidFormerBlock — one full dual-state layer.

Pipeline per layer:
    1. ClassicalAttention(c)               -> c'
    2. VoidAttention(v)                    -> v'  (+ entropy)
    3. InterferenceAttention(c', v')       -> i_pos, i_neg
    4. InterferenceEngine                  -> combined
    5. UncertaintyMemory(v') updates       -> v''
    6. AdaptiveCollapseEngine              -> y, τ
    7. DynamicRouter(c', v''_proj, y)      -> routed
    8. FFN + residuals
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .classical_attention import ClassicalAttention
from .collapse_engine import AdaptiveCollapseEngine
from .dynamic_routing import DynamicRouter
from .interference_attention import InterferenceAttention
from .interference_engine import InterferenceEngine
from .uncertainty_memory import UncertaintyMemory
from .void_attention import VoidAttention


class _FFN(nn.Module):
    def __init__(self, d: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoidFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_void: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        void_noise_std: float,
        collapse_temperature_init: float,
        use_uncertainty_memory: bool,
        use_dynamic_router: bool,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_void = d_void
        self.use_uncertainty_memory = use_uncertainty_memory
        self.use_dynamic_router = use_dynamic_router

        self.ln_c1 = nn.LayerNorm(d_model)
        self.ln_v1 = nn.LayerNorm(d_void)
        self.classical_attn = ClassicalAttention(d_model, n_heads, dropout)
        self.void_attn = VoidAttention(d_void, n_heads, dropout, noise_std=void_noise_std)

        self.ln_c2 = nn.LayerNorm(d_model)
        self.ln_v2 = nn.LayerNorm(d_void)
        self.interference_attn = InterferenceAttention(d_model, d_void, n_heads, dropout)
        self.interference = InterferenceEngine(d_model, d_void, dropout)

        self.memory = UncertaintyMemory(d_void) if use_uncertainty_memory else None

        self.collapse = AdaptiveCollapseEngine(d_model, temperature_init=collapse_temperature_init)
        self.v_to_c = nn.Linear(d_void, d_model, bias=False)
        self.router = DynamicRouter(d_model) if use_dynamic_router else None

        self.ln_c3 = nn.LayerNorm(d_model)
        self.ffn_c = _FFN(d_model, d_ff, dropout)
        self.ln_v3 = nn.LayerNorm(d_void)
        self.ffn_v = _FFN(d_void, d_ff, dropout)

    def forward(
        self,
        c: torch.Tensor,
        v: torch.Tensor,
        memory: torch.Tensor | None,
        depth_ratio: float,
        mode: str = "adaptive-collapse",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
        # 1-2. Self attentions on each stream
        c_attn, _ = self.classical_attn(self.ln_c1(c))
        v_attn, _, void_entropy = self.void_attn(self.ln_v1(v))
        c1 = c + c_attn
        v1 = v + v_attn

        # 3. Cross-stream interference
        i_pos, i_neg = self.interference_attn(self.ln_c2(c1), self.ln_v2(v1))

        # 4. Combine
        combined, coef_log = self.interference(c1, v1, i_pos, i_neg)

        # 5. Uncertainty memory
        new_memory = memory
        if self.memory is not None:
            v1, new_memory = self.memory(v1, memory)

        # 6. Adaptive collapse
        # mode → effective collapse strategy
        collapse_mode = {
            "deterministic": "hard",
            "probabilistic": "coexist",
            "interference-heavy": "soft",
            "adaptive-collapse": "soft",
            "latent-memory": "coexist",
        }.get(mode, "soft")
        y, tau = self.collapse(c1, combined, void_entropy.mean(dim=1), depth_ratio, collapse_mode)

        # 7. Dynamic routing
        v_in_c = self.v_to_c(v1)
        if self.router is not None:
            y, route_w = self.router(c1, v_in_c, y)
        else:
            route_w = None

        # 8. FFN + residuals
        c_out = y + self.ffn_c(self.ln_c3(y))
        v_out = v1 + self.ffn_v(self.ln_v3(v1))

        diag = {
            "void_entropy": void_entropy.detach(),       # (B, H, T)
            "tau": tau.detach(),                         # (B, T)
            "coefs": coef_log,                           # (B, T, 5)
            "route_weights": None if route_w is None else route_w.detach(),
        }
        return c_out, v_out, new_memory, diag
