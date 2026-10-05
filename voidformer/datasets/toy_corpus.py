"""Toy corpus — ambiguity-rich english fragments, no internet required.

Designed to exercise dual-state semantics: words like *cell, bank, bark, cold,
crane, light* recur in different contexts.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

_TEXT = """
The biological cell divided into two daughter cells under the microscope.
The prisoner returned to his cell after the recreation hour ended.
The battery cell powered the small flashlight for many hours.
She filled in each cell of the spreadsheet with quarterly figures.
The terrorist cell operated quietly within the city for many years.
A stem cell can differentiate into many specialised tissue types.

The river bank was eroded by the spring flood waters last march.
He deposited the cheque at the bank on the corner of fifth avenue.
The pilot rolled the aircraft into a steep bank during the turn.
The bank of switches controlled lighting across the entire stage.

The old crane lifted heavy steel beams onto the construction site.
A grey crane stood motionless at the edge of the shallow pond.

The dog barked at the stranger walking past the wooden gate.
The bark of the oak tree was rough beneath her bare hand.

Light from the candle barely reached the corner of the room.
The light feather drifted slowly through the still afternoon air.
She sees the world in light of recent scientific discoveries.

A cold wind blew across the empty parking lot at dawn.
He gave her a cold reply and walked away without looking back.
The patient came down with a bad cold during the long winter.

The patient remained patient while waiting for the diagnosis.
Time flies like an arrow; fruit flies like a banana.
""".strip()


class ToyCorpus(Dataset):
    def __init__(self, tokenizer, block_size: int = 64, repeats: int = 32) -> None:
        ids = tokenizer.encode(_TEXT * repeats)
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
        return {"input_ids": chunk[:-1], "targets": chunk[:-1]}  # next-token shift inside loss
