"""Lesson Dataset Loader for JSONL files containing prompt and answer pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class LessonDataset(Dataset):
    """Dataset loader for JSONL lesson files with prompt and answer fields."""

    def __init__(
        self,
        jsonl_file: str | Path,
        tokenizer: Any = None,
        max_seq_len: int = 128,
    ):
        self.jsonl_path = Path(jsonl_file)
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, str]] = []

        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    data = json.loads(line_str)
                    prompt = data.get("prompt", data.get("question", ""))
                    answer = data.get("answer", data.get("response", ""))
                    self.samples.append({"prompt": prompt, "answer": answer})

        if not self.samples:
            # Fallback dummy sample if empty or missing file
            self.samples = [{"prompt": "What is quantum superposition?", "answer": "Quantum superposition is state overlap."}]

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_text(self, text: str) -> List[int]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            return tokens
        else:
            # Default character-level encoding fallback
            return [ord(c) % 256 for c in text]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        full_text = f"{sample['prompt']}\n{sample['answer']}"
        tokens = self._encode_text(full_text)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        elif len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        tensor_tokens = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": tensor_tokens,
            "targets": tensor_tokens.clone(),
            "labels": tensor_tokens.clone(),
        }


def create_lesson_dataloader(
    jsonl_file: str | Path,
    tokenizer: Any = None,
    max_seq_len: int = 128,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LessonDataset(jsonl_file, tokenizer=tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
