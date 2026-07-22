from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hooks import ActivationCache
from model.layer_norm import RMSNorm
from model.attention import SelfAttention
from model.pos_encoding import rope_tables, apply_rope
from model.embedding import TimestepEmbedding

class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ------------------------------------------------------------------- the block
class HookedDiTBlock(nn.Module):
    """One pre-LN transformer block with adaLN-zero and two hook sites."""

    def __init__(self, layer_idx: int, d: int, n_heads: int, d_ff: int,
                 dropout: float, use_adaln: bool = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(d)
        self.attn = SelfAttention(d, n_heads, dropout)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, d_ff, dropout)

        if use_adaln:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d, bias=True))
            nn.init.zeros_(self.ada[1].weight)      # adaLN-ZERO: block starts as
            nn.init.zeros_(self.ada[1].bias)        # the identity function

    def forward(self, x, temb, cos, sin, key_padding_mask=None,
                steer: Optional[Callable] = None,
                cache: Optional[ActivationCache] = None,
                step: Optional[int] = None):

        # ---- Phase 2 hook sites (inert when both are None) ----------------
        if steer is not None:
            x = steer(x, self.layer_idx, step)
        if cache is not None:
            cache.record(self.layer_idx, x)
        # -------------------------------------------------------------------

        if self.use_adaln:
            s1, sc1, g1, s2, sc2, g2 = self.ada(temb).chunk(6, dim=-1)
            x = x + g1.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), s1, sc1), cos, sin, key_padding_mask)
            x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        else:
            x = x + self.attn(self.norm1(x), cos, sin, key_padding_mask)
            x = x + self.mlp(self.norm2(x))
        return x

# ------------------------------------------------------------------- the model
class MDLM(nn.Module):
    """Masked diffusion language model over SELFIES."""

    def __init__(self, vocab_size: int, cfg, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        d = cfg.d_model

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.temb = TimestepEmbedding(cfg.time_embed_dim, d)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            HookedDiTBlock(i, d, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.use_adaln)
            for i in range(cfg.n_layers)
        ])
        self.norm_out = RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight       # weight tying

        self.apply(self._init)
        for b in self.blocks:                        # re-zero after generic init
            if b.use_adaln:
                nn.init.zeros_(b.ada[1].weight)
                nn.init.zeros_(b.ada[1].bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                steer: Optional[Callable] = None,
                cache: Optional[ActivationCache] = None,
                step: Optional[int] = None) -> torch.Tensor:
        """x_t: (B, L) int64 token ids;  t: (B,) float in (0,1]."""
        B, L = x_t.shape
        h = self.drop(self.tok_emb(x_t))
        temb = self.temb(t)

        cos, sin = rope_tables(L, self.cfg.d_head, self.cfg.rope_base,
                               x_t.device, h.dtype)
        key_padding_mask = x_t.eq(self.pad_id)       # True = ignore

        for blk in self.blocks:
            h = blk(h, temb, cos, sin, key_padding_mask,
                    steer=steer, cache=cache, step=step)

        if cache is not None:                        # final residual, layer == n_layers
            cache.record(len(self.blocks), h)
        return self.head(self.norm_out(h))

    # ---- utilities -------------------------------------------------------
    def n_params(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)

    def param_breakdown(self) -> dict:
        d = {}
        d["embedding (tied)"] = self.tok_emb.weight.numel()
        d["time_embed"] = sum(p.numel() for p in self.temb.parameters())
        blk = self.blocks[0]
        d["per_block_attn"] = sum(p.numel() for p in blk.attn.parameters())
        d["per_block_mlp"] = sum(p.numel() for p in blk.mlp.parameters())
        d["per_block_adaln"] = (sum(p.numel() for p in blk.ada.parameters())
                                if blk.use_adaln else 0)
        d["n_blocks"] = len(self.blocks)
        d["total"] = self.n_params()
        return d

