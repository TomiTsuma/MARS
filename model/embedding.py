from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------ time conditioning
class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of continuous t in [0,1] -> MLP."""

    def __init__(self, dim: int, d_model: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)
