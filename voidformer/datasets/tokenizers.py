"""Tokenizers — char-level (offline default) and BPE (HuggingFace `tokenizers`)."""

from __future__ import annotations

from typing import Iterable, Protocol


class Tokenizer(Protocol):
    vocab_size: int
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class CharTokenizer:
    """ASCII byte-level char tokenizer (vocab_size=256, no training required)."""

    def __init__(self) -> None:
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))[: 1 << 20]

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class BPETokenizer:
    """Wrapper around HuggingFace `tokenizers` (BPE) — trains on supplied corpus.

    Falls back gracefully: if `tokenizers` is unavailable, raises ImportError
    so callers can switch to CharTokenizer.
    """

    def __init__(self, vocab_size: int = 8000) -> None:
        try:
            from tokenizers import Tokenizer as HFTok
            from tokenizers.models import BPE
        except ImportError as e:                                            # noqa: BLE001
            raise ImportError(
                "BPETokenizer requires `tokenizers` (`pip install tokenizers`)"
            ) from e
        self._HFTok = HFTok
        self._BPE = BPE
        self.vocab_size = vocab_size
        self._tok = None

    def train(self, texts: Iterable[str]) -> "BPETokenizer":
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        tok.train_from_iterator(texts, trainer)
        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
        return self

    def load_pretrained_gpt2(self) -> "BPETokenizer":
        """Use the pretrained GPT-2 BPE for offline-friendly default."""
        from transformers import GPT2TokenizerFast
        self._gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
        self.vocab_size = self._gpt2.vocab_size
        return self

    def encode(self, text: str) -> list[int]:
        if hasattr(self, "_gpt2"):
            return self._gpt2.encode(text)
        if self._tok is None:
            raise RuntimeError("BPETokenizer not trained — call .train(...) first")
        return self._tok.encode(text).ids

    def decode(self, ids) -> str:
        ids = list(ids)
        if hasattr(self, "_gpt2"):
            return self._gpt2.decode(ids, skip_special_tokens=True)
        return self._tok.decode(ids)


def build_tokenizer(cfg: dict, training_texts: Iterable[str] | None = None) -> Tokenizer:
    """Factory.

    cfg.tokenizer.type:
        char  -> CharTokenizer (zero-config, vocab=256)
        bpe   -> BPETokenizer; if `training_texts` supplied -> .train()
                 else -> tries gpt2 pretrained.
    """
    t = cfg["tokenizer"]["type"].lower()
    if t == "char":
        return CharTokenizer()
    if t == "bpe":
        bpe = BPETokenizer(vocab_size=cfg["tokenizer"].get("bpe_vocab_size", 8000))
        if training_texts is not None:
            return bpe.train(training_texts)
        return bpe.load_pretrained_gpt2()
    raise ValueError(f"unknown tokenizer type: {t}")
