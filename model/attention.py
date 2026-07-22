from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pos_encoding import apply_rope

class SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, L) True = attend
            attn_mask = (~key_padding_mask)[:, None, None, :]

        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,                       # bidirectional. Never change.
        )
        o = o.transpose(1, 2).reshape(B, L, D)
        return self.proj(o)
