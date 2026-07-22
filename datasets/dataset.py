"""
Dataset wrapper.

The batch carries three tensors, and keeping them distinct is what prevents
the two most common silent bugs in this pipeline:

    ids        (B, L) int64   token ids
    maskable   (B, L) bool    True where the forward process MAY mask
                              (i.e. real molecule tokens: not BOS/prefix/pad)
    valid      (B, L) bool    True where a token counts toward the per-token
                              NELBO denominator (identical to `maskable` here,
                              kept separate because they diverge if you ever
                              make the prefix a prediction target)

If pad or prefix positions leak into either mask, the loss looks great and the
model is learning nothing. See train/diagnostics.py.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SelfiesDataset(Dataset):
    def __init__(self, ids_array: np.ndarray, pad_id: int, n_struct: int):
        self.ids = torch.from_numpy(ids_array.astype(np.int64))
        self.pad_id = pad_id
        self.n_struct = n_struct       # 1 (BOS) + n_prefix

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, i: int):
        return self.ids[i]


def make_masks(ids: torch.Tensor, pad_id: int, n_struct: int):
    """Returns (maskable, valid) boolean tensors of shape (B, L).

    LENGTH MODELLING — read this before "optimising" it away.
    -------------------------------------------------------
    Padding IS maskable and IS a loss target. This looks wasteful and is the
    opposite of autoregressive practice, where an explicit EOS plus a causal
    decoder handles length. A masked diffusion LM has no such mechanism: at
    sampling time it starts from an all-[MASK] body of fixed width and must
    decide for itself where the molecule stops.

    If [PAD] is excluded from the loss the model never learns to emit it, and
    every sample fills the full body with molecule tokens. Training accuracy
    still looks perfect, because the failure appears only at sampling. This is
    exactly the bug GATE 2 exists to catch.

    Consequence to keep in mind: PAD positions are easy predictions, so both
    accuracy and the per-token NELBO are optimistic relative to a molecule-only
    denominator. Track `real_token_acc` in the trainer for the honest number.
    """
    L = ids.shape[-1]
    pos = torch.arange(L, device=ids.device)
    is_struct = (pos < n_struct).unsqueeze(0).expand_as(ids)
    maskable = ~is_struct                 # includes PAD: the model learns length
    valid = maskable.clone()
    return maskable, valid


def collate(batch, pad_id: int, n_struct: int):
    ids = torch.stack(batch, dim=0)
    maskable, valid = make_masks(ids, pad_id, n_struct)
    return {"ids": ids, "maskable": maskable, "valid": valid}


def make_loader(ids_array: np.ndarray, pad_id: int, n_struct: int,
                batch_size: int, shuffle: bool = True,
                num_workers: int = 4, seed: int = 0) -> DataLoader:
    ds = SelfiesDataset(ids_array, pad_id, n_struct)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
        num_workers=num_workers, pin_memory=True, generator=g,
        collate_fn=lambda b: collate(b, pad_id, n_struct),
    )
