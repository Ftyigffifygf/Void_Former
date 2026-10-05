"""Sanity experiment — instantiates a tiny model, runs a forward pass on a
short prompt, and dumps all visualisation plots to the experiment output dir.

Usage:
    python -m voidformer.experiments.sanity_run --config voidformer/configs/tiny.yaml
"""

from __future__ import annotations

import argparse
import os

import torch

from ..datasets import build_tokenizer
from ..models import VoidFormerModel
from ..utils import load_config, set_seed, get_logger
from ..visualization import (
    plot_collapse_trajectory,
    plot_entropy_heatmap,
    plot_geodesic_distribution,
    plot_geodesic_path,
    plot_interference_coefs,
    plot_route_weights,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--prompt", default="The cell divided into")
    p.add_argument("--mode", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    log = get_logger("voidformer.sanity")

    tok = build_tokenizer(cfg)
    vocab = tok.vocab_size
    cfg["model"]["vocab_size"] = vocab

    model = VoidFormerModel(**cfg["model"])
    if args.mode:
        model.set_mode(args.mode)
    log.info("VoidFormer params: %d", model.num_params())

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    ids = ids[:, : model.max_seq_len]
    out = model(ids, return_diagnostics=True)
    log.info(
        "logits=%s classical=%s void=%s",
        tuple(out.logits.shape),
        tuple(out.classical_states.shape),
        tuple(out.void_states.shape),
    )

    outdir = cfg["experiment"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)

    plot_entropy_heatmap(out.diagnostics, os.path.join(outdir, "entropy_heatmap.png"))
    plot_collapse_trajectory(out.diagnostics, os.path.join(outdir, "collapse_trajectory.png"))
    plot_route_weights(out.diagnostics, os.path.join(outdir, "route_weights.png"))
    plot_interference_coefs(out.diagnostics, os.path.join(outdir, "interference_coefs.png"))
    plot_geodesic_path(
        out.geodesic_segments,
        out.geodesic_distance,
        os.path.join(outdir, "geodesic_path.png"),
    )
    plot_geodesic_distribution(
        out.geodesic_distance,
        os.path.join(outdir, "geodesic_distribution.png"),
    )
    if out.geodesic_distance is not None:
        gd = out.geodesic_distance.detach()
        log.info(
            "geodesic d(I,U) per token: mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(gd.mean()),
            float(gd.std()),
            float(gd.min()),
            float(gd.max()),
        )
    log.info("plots written to %s", outdir)


if __name__ == "__main__":
    main()
