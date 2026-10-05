"""Evaluation Script for VoidFormer Checkpoints on Held-Out Exam Data."""

from __future__ import annotations

import argparse
import json
import os
import yaml
import torch

from voidformer.harness.model_factory import create_model
from voidformer.utils.checkpoint import load_checkpoint
from voidformer.datasets.lesson_dataset import LessonDataset, create_lesson_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VoidFormer Model Checkpoint on Exam JSONL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--exam-file", type=str, required=True, help="Path to exam .jsonl file")
    parser.add_argument("--config", type=str, default="voidformer/configs/tiny.yaml", help="Path to config")
    parser.add_argument("--model-type", type=str, choices=["quantum", "classical"], default="quantum", help="Model type")
    parser.add_argument("--output-json", type=str, default="eval_results.json", help="Path to output JSON")
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint_path: str,
    exam_file: str,
    config_path: str = "voidformer/configs/tiny.yaml",
    model_type: str = "quantum",
    output_json: str = "eval_results.json",
) -> dict:
    if not os.path.exists(config_path):
        config_path = os.path.join("voidformer", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    model = create_model(
        model_type=model_type,
        vocab_size=model_cfg.get("vocab_size", 256),
        d_model=model_cfg.get("d_model", 128),
        d_void=model_cfg.get("d_void", 128),
        n_layers=model_cfg.get("n_layers", 2),
        n_heads=model_cfg.get("n_heads", 4),
        d_ff=model_cfg.get("d_ff", 256),
        max_seq_len=model_cfg.get("max_seq_len", 128),
        use_vqc_layer=model_cfg.get("use_vqc_layer", True),
    )

    if os.path.exists(checkpoint_path):
        step, seed, loaded_cfg = load_checkpoint(checkpoint_path, model=model)
    else:
        step = 0

    model.eval()

    dataloader = create_lesson_dataloader(
        jsonl_file=exam_file,
        max_seq_len=model_cfg.get("max_seq_len", 128),
        batch_size=4,
        shuffle=False,
    )

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"]
            targets = batch["targets"]

            output = model(ids)
            logits = output.logits if hasattr(output, "logits") else output

            # Internal shifting convention: logits[:, :-1] vs targets[:, 1:]
            if logits.ndim == 3 and targets.ndim == 2 and logits.size(1) == targets.size(1) and logits.size(1) > 1:
                logits_shift = logits[:, :-1, :].contiguous()
                targets_shift = targets[:, 1:].contiguous()
            else:
                logits_shift = logits
                targets_shift = targets

            loss = criterion(logits_shift.reshape(-1, logits_shift.size(-1)), targets_shift.reshape(-1))
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    perplexity = float(torch.exp(torch.tensor(avg_loss)).item())

    results = {
        "checkpoint": checkpoint_path,
        "exam_file": exam_file,
        "model_type": model_type,
        "step": step,
        "loss": avg_loss,
        "perplexity": perplexity,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"Evaluation Complete | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    return results


def main():
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        exam_file=args.exam_file,
        config_path=args.config,
        model_type=args.model_type,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
