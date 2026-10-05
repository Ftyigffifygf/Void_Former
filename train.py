"""Training Entrypoint for VoidFormer Model Suite."""

from __future__ import annotations

import argparse
import os
import sys
import yaml
import torch

from voidformer.harness.model_factory import create_model
from voidformer.training.trainer import Trainer
from voidformer.training.losses import VoidFormerLosses, LossWeights
from voidformer.harness.data import create_dataloader
from voidformer.datasets.lesson_dataset import create_lesson_dataloader
from voidformer.utils.seed import set_seed
from voidformer.utils.checkpoint import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train VoidFormer Language Model")
    parser.add_argument("--config", type=str, default="voidformer/configs/tiny.yaml", help="Path to YAML config")
    parser.add_argument("--model-type", type=str, choices=["quantum", "classical"], default="quantum", help="Model type")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--lesson-file", type=str, default=None, help="Path to JSONL lesson file for training")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config file
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join("voidformer", args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir

    model_type = args.model_type
    model_cfg = cfg.get("model", {})

    print(f"Initializing {model_type.upper()} VoidFormer Model...")
    model = create_model(
        model_type=model_type,
        vocab_size=model_cfg.get("vocab_size", 256),
        d_model=model_cfg.get("d_model", 128),
        d_void=model_cfg.get("d_void", 128),
        n_layers=model_cfg.get("n_layers", 2),
        n_heads=model_cfg.get("n_heads", 4),
        d_ff=model_cfg.get("d_ff", 256),
        max_seq_len=model_cfg.get("max_seq_len", 128),
        use_superposition_thinking=model_cfg.get("use_superposition_thinking", True),
        thinking_steps=model_cfg.get("thinking_steps", 4),
        use_quantum_moe=model_cfg.get("use_quantum_moe", False),
        num_experts=model_cfg.get("num_experts", 4),
        top_k_experts=model_cfg.get("top_k_experts", 2),
        use_quantum_token_embedder=model_cfg.get("use_quantum_token_embedder", False),
    )

    losses_cfg = cfg.get("losses", {})
    loss_fn = VoidFormerLosses(LossWeights(**{k: v for k, v in losses_cfg.items() if k in LossWeights.__dataclass_fields__}))

    if args.lesson_file and os.path.exists(args.lesson_file):
        print(f"Loading lesson dataset from {args.lesson_file}...")
        dataloader = create_lesson_dataloader(
            jsonl_file=args.lesson_file,
            max_seq_len=model_cfg.get("max_seq_len", 128),
            batch_size=cfg.get("dataset", {}).get("batch_size", 4),
        )
    else:
        dataloader = create_dataloader(
            vocab_size=model_cfg.get("vocab_size", 256),
            seq_len=model_cfg.get("max_seq_len", 128),
            batch_size=cfg.get("dataset", {}).get("batch_size", 4),
            num_samples=100,
        )

    trainer = Trainer(model=model, loss_fn=loss_fn, train_loader=dataloader, cfg=cfg)

    print(f"Starting training run ({model_type.upper()}) for {cfg['training']['total_steps']} steps...")
    result = trainer.fit()

    out_dir = cfg.get("experiment", {}).get("output_dir", "experiments/ckpt")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "model_latest.pt")
    save_checkpoint(ckpt_path, model=model, optimizer=trainer.optim, config=cfg, step=cfg['training']['total_steps'], seed=args.seed)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
