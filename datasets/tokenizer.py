"""
SELFIES tokenizer.

Vocabulary layout is IRREVERSIBLE — it is baked into every checkpoint:

    0        [PAD]     padding; excluded from loss and is masked and unmasked
    1        [MASK]    the absorbing state
    2        [BOS]     sequence start
    3..3+P-1 [P0]..    STRUCTURAL PREFIX. Never masked, never a loss target.
                       These are the position-invariant extraction sites that
                       Phase 2 steering depends on. They cannot be added later.
    3+P..    SELFIES symbols, sorted, taken from the TRAINING corpus alphabet.

Encoded layout of every sequence, padded to `max_len`:

    [BOS] [P0] [P1] ... [P{P-1}] <molecule tokens> [PAD] [PAD] ...
     ^--------- never masked ---------^  ^-- maskable --^  ^- never masked -^
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import selfies as sf

PAD = "[PAD]"
MASK = "[MASK]"
BOS = "[BOS]"

@dataclass
class SelfiesTokenizer:
    symbols: List[str]
    n_prefix: int = 8
    max_len: int = 128

    # ---- construction ----------------------------------------------------
    @staticmethod
    def build(
        selfies_strings: Iterable[str],
        n_prefix: int = 8,
        max_len: int = 128
    ) -> "SelfiesTokenizer":
        """Build the alphabet from TRAINING data only. Using the full dataset
        here is a leakage bug that is invisible until someone asks."""
        alphabet = sf.get_alphabet_from_selfies(selfies_strings)
        return SelfiesTokenizer(sorted(alphabet), n_prefix=n_prefix, max_len=max_len)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"symbols": self.symbols,
                       "n_prefix": self.n_prefix,
                       "max_len": self.max_len}, f)

    @staticmethod
    def load(path: str) -> "SelfiesTokenizer":
        with open(path) as f:
            d = json.load(f)
        return SelfiesTokenizer(**d)

    # ---- vocabulary ------------------------------------------------------
    def __post_init__(self):
        self.prefix_tokens = [f"[P{i}]" for i in range(self.n_prefix)]
        self.itos: List[str] = [PAD, MASK, BOS] + self.prefix_tokens + list(self.symbols)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

        self.pad_id = self.stoi[PAD]
        self.mask_id = self.stoi[MASK]
        self.bos_id = self.stoi[BOS]
        self.prefix_ids = [self.stoi[t] for t in self.prefix_tokens]

        # Positions 0..n_prefix are BOS + prefix: structural, never masked.
        self.n_struct = 1 + self.n_prefix

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ---- encode / decode -------------------------------------------------
    def encode(self, selfies_str: str) -> Optional[List[int]]:
        """SELFIES string -> padded id list. Returns None if it does not fit."""
        toks = list(sf.split_selfies(selfies_str))
        if len(toks) + self.n_struct > self.max_len:
            return None
        try:
            body = [self.stoi[t] for t in toks]
        except KeyError:
            return None                      # OOV symbol: not in training alphabet
        ids = [self.bos_id] + self.prefix_ids + body
        ids += [self.pad_id] * (self.max_len - len(ids))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """id list -> SELFIES string.

        Decoding TRUNCATES at the first [PAD] after the structural prefix.
        [PAD] is the model's learned terminator (see data/dataset.make_masks),
        so anything after it is not part of the molecule. Skipping pads instead
        of truncating would splice disconnected fragments together and inflate
        apparent validity while producing chemistry the model did not intend.

        A [MASK] surviving to decode time indicates a sampler bug; we stop there
        rather than silently emitting a different molecule.
        """
        out = []
        for j, i in enumerate(ids):
            if j < self.n_struct:                 # BOS + prefix
                continue
            tok = self.itos[int(i)]
            if tok in (PAD, MASK):
                break                             # learned terminator
            if tok == BOS or tok in self.prefix_tokens:
                continue
            out.append(tok)
        return "".join(out)

    def decode_smiles(self, ids: Sequence[int]) -> Optional[str]:
        try:
            return sf.decoder(self.decode(ids))
        except Exception:
            return None

    # ---- masks -----------------------------------------------------------
    def structural_mask(self, ids) -> "list[bool]":
        """True where the position is BOS/prefix (never masked, never a target)."""
        return [j < self.n_struct for j in range(len(ids))]
