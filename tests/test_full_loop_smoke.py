"""Full System Smoke Test: Train, Checkpoint, Evaluate, Classroom Round, and FastAPI Serve Endpoint."""

from __future__ import annotations

import os
import json
import yaml
import torch
import pytest
from fastapi.testclient import TestClient

from voidformer.harness.model_factory import create_model
from voidformer.utils.checkpoint import save_checkpoint, load_checkpoint
from voidformer.datasets.lesson_dataset import LessonDataset, create_lesson_dataloader
from voidformer.training.trainer import Trainer
from voidformer.training.losses import VoidFormerLosses, LossWeights
from voidformer.training.distillation import DistillationLoss
from evaluate import evaluate_checkpoint
from classroom import run_classroom_round
from serve import app, load_server_model


def test_full_loop_smoke(tmp_path):
    # 1. Config & Paths
    ckpt_path = tmp_path / "model_test.pt"
    lesson_jsonl = tmp_path / "lessons.jsonl"
    exam_jsonl = tmp_path / "exam.jsonl"
    eval_json = tmp_path / "eval_out.json"

    # Create dummy JSONL files
    with open(lesson_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "What is quantum superposition?", "answer": "State overlap."}) + "\n")

    with open(exam_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "What is quantum superposition?", "answer": "State overlap."}) + "\n")

    # 2. Model Creation
    model = create_model(model_type="quantum", vocab_size=256, d_model=64, n_heads=2, use_vqc_layer=False)
    dataloader = create_lesson_dataloader(jsonl_file=lesson_jsonl, max_seq_len=32, batch_size=2)

    cfg = {
        "training": {"lr": 1e-3, "total_steps": 2, "device": "cpu", "log_every": 1},
        "experiment": {"output_dir": str(tmp_path), "tensorboard": False},
        "model": {"vocab_size": 256, "d_model": 64, "n_heads": 2, "use_vqc_layer": False},
    }

    loss_fn = VoidFormerLosses({})
    trainer = Trainer(model=model, loss_fn=loss_fn, train_loader=dataloader, cfg=cfg)

    # 3. Train for 2 steps
    res = trainer.fit(max_steps=2)
    assert res["elapsed"] > 0

    # 4. Save & Load Checkpoint
    save_checkpoint(ckpt_path, model=model, optimizer=trainer.optim, config=cfg, step=2, seed=1234)
    assert os.path.exists(ckpt_path)

    step, seed, loaded_cfg = load_checkpoint(ckpt_path, model)
    assert step == 2

    # 5. Evaluate Checkpoint
    # Save a test config
    test_cfg_path = tmp_path / "test_cfg.yaml"
    with open(test_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    eval_res = evaluate_checkpoint(
        checkpoint_path=str(ckpt_path),
        exam_file=str(exam_jsonl),
        config_path=str(test_cfg_path),
        model_type="quantum",
        output_json=str(eval_json),
    )
    assert "loss" in eval_res
    assert "perplexity" in eval_res

    # 6. Distillation Loss
    distill = DistillationLoss()
    student_logits = torch.randn(2, 8, 256)
    targets = torch.randint(0, 256, (2, 8))
    teacher_logits = torch.randn(2, 8, 256)
    d_loss = distill(student_logits, targets, teacher_logits)
    assert d_loss.item() > 0.0

    # 7. Classroom Round Execution
    class_summary = run_classroom_round(
        round_idx=1,
        teacher_ckpt=str(ckpt_path),
        student_model_type="quantum",
        config_path=str(test_cfg_path),
        rounds_dir=str(tmp_path / "runs"),
        steps_per_round=2,
    )
    assert class_summary["round"] == 1

    # 8. Serve Endpoint (FastAPI) Test
    load_server_model(checkpoint_path=str(ckpt_path), config_path=str(test_cfg_path), model_type="quantum")
    client = TestClient(app)

    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200
    assert "data" in models_resp.json()

    chat_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "quantum-voidformer",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        },
    )
    assert chat_resp.status_code == 200
    assert "choices" in chat_resp.json()
