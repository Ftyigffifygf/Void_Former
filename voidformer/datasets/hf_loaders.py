"""HuggingFace dataset loaders for WikiText / PubMed / arXiv.

These require network access and the `datasets` library. The `toy` corpus
is always available offline.
"""

from __future__ import annotations


import torch
from torch.utils.data import Dataset

from .toy_corpus import ToyCorpus


class _BlockDataset(Dataset):
    """Concat-and-chunk style LM dataset."""

    def __init__(self, ids: list[int], block_size: int) -> None:
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, (self.ids.size(0) - 1) // self.block_size)

    def __getitem__(self, idx: int) -> dict:
        s = idx * self.block_size
        e = s + self.block_size
        chunk = self.ids[s : e + 1]
        if chunk.size(0) < self.block_size + 1:
            pad = torch.zeros(self.block_size + 1 - chunk.size(0), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return {"input_ids": chunk[:-1], "targets": chunk[:-1]}


def _load_hf(name: str, hf_name: str | None) -> list[str]:
    from datasets import load_dataset

    if name == "wikitext":
        ds = load_dataset("wikitext", hf_name or "wikitext-2-raw-v1", split="train")
        return [x for x in ds["text"] if x and x.strip()]
    if name == "pubmed":
        # Public mirror; subset for speed.
        ds = load_dataset("ccdv/pubmed-summarization", split="train[:5000]")
        return [x for x in ds["article"] if x and x.strip()]
    if name == "arxiv":
        ds = load_dataset("ccdv/arxiv-summarization", split="train[:5000]")
        return [x for x in ds["article"] if x and x.strip()]
    raise ValueError(f"unknown HF dataset: {name}")


def build_dataset(cfg: dict, tokenizer) -> Dataset:
    d = cfg["dataset"]
    name = d["type"].lower()
    block = d["block_size"]
    if name == "toy":
        return ToyCorpus(tokenizer, block_size=block)
    texts = _load_hf(name, d.get("hf_name"))
    encoded: list[int] = []
    for txt in texts:
        encoded.extend(tokenizer.encode(txt))
    return _BlockDataset(encoded, block_size=block)
