"""Classroom Pipeline — Teacher Generation, Filtering, Student Distillation, Exam Evaluation & Round Logging."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import yaml
import torch

from voidformer.harness.model_factory import create_model
from voidformer.utils.checkpoint import save_checkpoint, load_checkpoint
from voidformer.datasets.lesson_dataset import create_lesson_dataloader
from voidformer.training.trainer import Trainer
from voidformer.training.losses import VoidFormerLosses, LossWeights
from voidformer.training.distillation import DistillationLoss
from evaluate import evaluate_checkpoint


def generate_teacher_lessons(teacher_model: torch.nn.Module, output_jsonl: str, num_lessons: int = 20):
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    teacher_model.eval()

    prompts = [
        "What is quantum superposition?",
        "Explain quantum entanglement.",
        "How does quantum interference work?",
        "Describe Hilbert space reasoning.",
        "What is Born rule measurement collapse?",
    ]

    samples = []
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i in range(num_lessons):
            prompt = prompts[i % len(prompts)]
            prompt_tokens = [ord(c) % 256 for c in prompt]
            ids = torch.tensor([prompt_tokens], dtype=torch.long)

            with torch.no_grad():
                if hasattr(teacher_model, "generate"):
                    gen_ids = teacher_model.generate(ids, max_new_tokens=24)
                else:
                    gen_ids = ids

            gen_text = "".join([chr(tok.item() % 128) for tok in gen_ids[0]])
            answer = gen_text[len(prompt):].strip()

            # Quality Filter: keep non-empty answers
            if len(answer) >= 3:
                obj = {"prompt": prompt, "answer": answer}
                samples.append(obj)
                f.write(json.dumps(obj) + "\n")

    if not samples:
        # If model is untrained, fallback to sample lesson dataset
        samples = [
            {"prompt": "What is quantum superposition?", "answer": "State overlap in Hilbert space."},
            {"prompt": "Explain quantum entanglement.", "answer": "Non-local correlation between qubits."},
        ]
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    return len(samples)


def create_held_out_exam_jsonl(exam_jsonl: str):
    """Creates held-out exam dataset with prompts distinct from lesson prompts."""
    os.makedirs(os.path.dirname(os.path.abspath(exam_jsonl)), exist_ok=True)
    exam_data = [
        {"prompt": "Define decoherence time T2*", "answer": "Phase relaxation time constant."},
        {"prompt": "What is quantum gate unitarity?", "answer": "Operator satisfying U dagger U equals I."},
        {"prompt": "How does state fidelity measure overlap?", "answer": "Magnitude squared inner product of statevectors."},
        {"prompt": "Explain Grover diffusion operator.", "answer": "Reflection about average state amplitude."},
    ]
    with open(exam_jsonl, "w", encoding="utf-8") as f:
        for item in exam_data:
            f.write(json.dumps(item) + "\n")


class DistillationLossWrapper(torch.nn.Module):
    """Loss wrapper combining VoidFormerLosses with DistillationLoss."""

    def __init__(self, base_loss_fn: VoidFormerLosses, distill_loss_fn: DistillationLoss):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.distill_loss_fn = distill_loss_fn

    def forward(self, out: Any, targets: torch.Tensor, embedding_weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
        base_loss, log = self.base_loss_fn(out, targets, embedding_weight)
        logits = out.logits if hasattr(out, "logits") else out
        d_loss = self.distill_loss_fn(logits, targets)
        total = base_loss + d_loss
        log["loss/distillation"] = d_loss.detach()
        log["loss/total"] = total.detach()
        return total, log


def run_classroom_round(
    round_idx: int,
    teacher_ckpt: str | None,
    student_model_type: str = "quantum",
    config_path: str = "voidformer/configs/tiny.yaml",
    rounds_dir: str = "runs",
    steps_per_round: int = 10,
    lessons_file: str | None = None,
    use_distillation: bool = True,
):
    round_dir = os.path.join(rounds_dir, f"round_{round_idx}")
    os.makedirs(round_dir, exist_ok=True)

    lesson_file = os.path.join(round_dir, "lessons.jsonl")
    exam_file = os.path.join(round_dir, "exam.jsonl")
    student_ckpt = os.path.join(round_dir, "student_latest.pt")
    eval_json = os.path.join(round_dir, "exam_results.json")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})

    # 1. Lesson dataset handling
    if lessons_file and os.path.exists(lessons_file):
        shutil.copy(lessons_file, lesson_file)
        with open(lesson_file, "r", encoding="utf-8") as f:
            num_generated = sum(1 for line in f if line.strip())
    else:
        teacher_model = create_model(
            model_type="quantum",
            vocab_size=model_cfg.get("vocab_size", 256),
            d_model=model_cfg.get("d_model", 128),
            n_layers=model_cfg.get("n_layers", 2),
            n_heads=model_cfg.get("n_heads", 4),
            d_ff=model_cfg.get("d_ff", 256),
            max_seq_len=model_cfg.get("max_seq_len", 128),
            use_vqc_layer=model_cfg.get("use_vqc_layer", True),
        )
        if teacher_ckpt and os.path.exists(teacher_ckpt):
            load_checkpoint(teacher_ckpt, teacher_model)

        num_generated = generate_teacher_lessons(teacher_model, lesson_file, num_lessons=20)

    # 2. Create held-out exam dataset
    create_held_out_exam_jsonl(exam_file)

    # 3. Train Student Model
    student_model = create_model(
        model_type=student_model_type,
        vocab_size=model_cfg.get("vocab_size", 256),
        d_model=model_cfg.get("d_model", 128),
        n_layers=model_cfg.get("n_layers", 2),
        n_heads=model_cfg.get("n_heads", 4),
        d_ff=model_cfg.get("d_ff", 256),
        max_seq_len=model_cfg.get("max_seq_len", 128),
        use_vqc_layer=model_cfg.get("use_vqc_layer", True),
    )

    dataloader = create_lesson_dataloader(
        jsonl_file=lesson_file,
        max_seq_len=model_cfg.get("max_seq_len", 128),
        batch_size=4,
    )

    losses_cfg = cfg.get("losses", {})
    base_loss_fn = VoidFormerLosses(LossWeights(**{k: v for k, v in losses_cfg.items() if k in LossWeights.__dataclass_fields__}))

    if use_distillation:
        loss_fn = DistillationLossWrapper(base_loss_fn, DistillationLoss())
    else:
        loss_fn = base_loss_fn

    cfg["training"]["total_steps"] = steps_per_round
    trainer = Trainer(model=student_model, loss_fn=loss_fn, train_loader=dataloader, cfg=cfg)
    trainer.fit(max_steps=steps_per_round)

    save_checkpoint(student_ckpt, model=student_model, optimizer=trainer.optim, config=cfg, step=steps_per_round)

    # 4. Held-out Exam Evaluation
    eval_results = evaluate_checkpoint(
        checkpoint_path=student_ckpt,
        exam_file=exam_file,
        config_path=config_path,
        model_type=student_model_type,
        output_json=eval_json,
    )

    summary = {
        "round": round_idx,
        "student_model_type": student_model_type,
        "lessons_count": num_generated,
        "student_checkpoint": student_ckpt,
        "exam_results": eval_results,
    }

    with open(os.path.join(round_dir, "round_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"--- Classroom Round {round_idx} Complete ({student_model_type.upper()}) ---")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Classroom Self-Play Training Loop")
    parser.add_argument("--rounds", type=int, default=2, help="Number of classroom rounds")
    parser.add_argument("--model-type", type=str, choices=["quantum", "classical"], default="quantum", help="Student model type")
    parser.add_argument("--lessons-file", type=str, default=None, help="Optional external teacher lesson JSONL file")
    parser.add_argument("--config", type=str, default="voidformer/configs/tiny.yaml", help="Path to config")
    parser.add_argument("--steps-per-round", type=int, default=10, help="Training steps per round")
    args = parser.parse_args()

    teacher_ckpt = None
    for r in range(1, args.rounds + 1):
        summary = run_classroom_round(
            round_idx=r,
            teacher_ckpt=teacher_ckpt,
            student_model_type=args.model_type,
            config_path=args.config,
            steps_per_round=args.steps_per_round,
            lessons_file=args.lessons_file,
        )
        teacher_ckpt = summary["student_checkpoint"]


if __name__ == "__main__":
    main()
