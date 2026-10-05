"""Inference Entrypoint for VoidFormer Model Suite."""

from __future__ import annotations

import argparse
import os
import yaml
import torch

from voidformer.harness.model_factory import create_model
from voidformer.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="VoidFormer Inference & Token Generation")
    parser.add_argument("--config", type=str, default="voidformer/configs/tiny.yaml", help="Path to config")
    parser.add_argument("--model-type", type=str, choices=["quantum", "classical"], default="quantum", help="Model type")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", type=str, default="quantum computing enables", help="Input prompt string")
    parser.add_argument("--max-tokens", type=int, default=32, help="Number of new tokens to generate")
    parser.add_argument("--use-quantum", action="store_true", default=True, help="Use quantum processing in generate")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join("voidformer", args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    model = create_model(
        model_type=args.model_type,
        vocab_size=model_cfg.get("vocab_size", 256),
        d_model=model_cfg.get("d_model", 128),
        d_void=model_cfg.get("d_void", 128),
        n_layers=model_cfg.get("n_layers", 2),
        n_heads=model_cfg.get("n_heads", 4),
        d_ff=model_cfg.get("d_ff", 256),
        max_seq_len=model_cfg.get("max_seq_len", 128),
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        load_checkpoint(args.checkpoint, model=model)

    model.eval()

    # Encode character-level or simple tokens
    prompt_tokens = [ord(c) % 256 for c in args.prompt]
    ids = torch.tensor([prompt_tokens], dtype=torch.long)

    print(f"Prompt: '{args.prompt}'")
    print(f"Generating {args.max_tokens} tokens...")

    if hasattr(model, "generate"):
        if args.model_type == "quantum":
            out_ids = model.generate(ids, max_new_tokens=args.max_tokens, use_quantum=args.use_quantum)
        else:
            out_ids = model.generate(ids, max_new_tokens=args.max_tokens)
    else:
        out_ids = ids

    gen_chars = "".join([chr(tok.item() % 128) for tok in out_ids[0]])
    print(f"Generated Output:\n{gen_chars}")


if __name__ == "__main__":
    main()
