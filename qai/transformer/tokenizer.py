"""Lightweight character and subword tokenizer for sequence modeling."""

from __future__ import annotations
from typing import List, Dict, Optional, Union
import torch


class SimpleTokenizer:
    """Character/token level tokenizer with special token handling."""

    def __init__(self, vocab: Optional[List[str]] = None) -> None:
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        if vocab is None:
            # Default small printable ASCII charset
            chars = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [" ", "0", "1", "2", "3"]
            self.vocab = self.special_tokens + sorted(list(set(chars)))
        else:
            self.vocab = self.special_tokens + [tok for tok in vocab if tok not in self.special_tokens]

        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = list(text)
        ids = [self.token_to_id.get(ch, self.token_to_id[self.unk_token]) for ch in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for i in ids:
            tok = self.id_to_token.get(int(i), self.unk_token)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            chars.append(tok)
        return "".join(chars)
