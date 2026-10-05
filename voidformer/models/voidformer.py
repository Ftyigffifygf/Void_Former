"""VoidFormerModel — full dual-state transformer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..layers import DualEmbedding, RiemannianGeodesicTracker, VoidFormerBlock


@dataclass
class VoidFormerOutput:
    logits: torch.Tensor                       # (B, T, V)
    classical_states: torch.Tensor             # (B, T, d_model) final
    void_states: torch.Tensor                  # (B, T, d_void)  final
    diagnostics: list[dict] = field(default_factory=list)
    geodesic_distance: torch.Tensor | None = None
    geodesic_segments: list = field(default_factory=list)


class VoidFormerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        d_void: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        void_noise_std: float = 0.05,
        coupling_init: float = 0.1,
        collapse_temperature_init: float = 1.0,
        use_uncertainty_memory: bool = True,
        use_dynamic_router: bool = True,
        use_geodesic_tracker: bool = True,
        geodesic_rank: int = 16,
        mode: str = "adaptive-collapse",
        tie_embeddings: bool = True,
        **_unused,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_void = d_void
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.mode = mode

        self.embed = DualEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            d_void=d_void,
            max_seq_len=max_seq_len,
            coupling_init=coupling_init,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            VoidFormerBlock(
                d_model=d_model,
                d_void=d_void,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                void_noise_std=void_noise_std,
                collapse_temperature_init=collapse_temperature_init,
                use_uncertainty_memory=use_uncertainty_memory,
                use_dynamic_router=use_dynamic_router,
            )
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.classical_emb.weight

        self.geodesic_tracker = (
            RiemannianGeodesicTracker(d_model, rank=geodesic_rank)
            if use_geodesic_tracker
            else None
        )

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def forward(
        self,
        ids: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> VoidFormerOutput:
        assert ids.size(1) <= self.max_seq_len, (
            f"sequence length {ids.size(1)} > max {self.max_seq_len}"
        )
        c, v = self.embed(ids)
        memory = (
            self.blocks[0].memory.init_state(v.size(0), v.size(1), self.d_void, v.device)
            if (self.blocks[0].memory is not None)
            else None
        )
        diagnostics: list[dict] = []
        if self.geodesic_tracker is not None:
            self.geodesic_tracker.reset()
        prev_c = c
        for i, block in enumerate(self.blocks):
            depth_ratio = (i + 1) / self.n_layers
            c, v, memory, diag = block(c, v, memory, depth_ratio, mode=self.mode)
            if self.geodesic_tracker is not None:
                # Per-layer Riemannian step.
                # Per-token entropy (mean over heads) drives the conformal factor κ.
                token_entropy = diag["void_entropy"].mean(dim=1)  # (B, T)
                seg = self.geodesic_tracker.step(prev_c, c, token_entropy)
                diag["geodesic_step"] = seg.detach()
            prev_c = c
            if return_diagnostics:
                diagnostics.append(diag)

        c = self.ln_f(c)
        logits = self.lm_head(c)

        geo_total: torch.Tensor | None = None
        geo_segs: list[torch.Tensor] = []
        if self.geodesic_tracker is not None:
            geo_total, geo_segs = self.geodesic_tracker.finalize()

        return VoidFormerOutput(
            logits=logits,
            classical_states=c,
            void_states=v,
            diagnostics=diagnostics,
            geodesic_distance=geo_total,
            geodesic_segments=geo_segs,
        )

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            crop = ids[:, -self.max_seq_len:]
            out = self.forward(crop)
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
