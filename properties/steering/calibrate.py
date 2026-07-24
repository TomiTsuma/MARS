"""
Coefficient calibration.

Two jobs:

  1. Persist and reload per-layer residual-stream standard deviations. Every
     alpha in Phase 2 is expressed in these units, so they are as much a part
     of a reported result as the direction itself.

  2. Provide the piecewise fallback. If dose-response turns out to be monotone
     but non-linear, a calibrated mapping from target property value to alpha
     preserves usability while reporting the underlying non-linearity honestly
     rather than hiding it behind a fitted line.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch


# ------------------------------------------------------- residual statistics
def save_residual_stats(std: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"std": std.detach().float().cpu()}, path)


def load_residual_stats(path: str):
    """Returns an object exposing .std indexable by layer."""
    blob = torch.load(path, map_location="cpu")
    std = blob["std"] if isinstance(blob, dict) else blob

    class _Stats:
        pass
    s = _Stats()
    s.std = std
    return s


@torch.no_grad()
def fit_residual_stats(model, loader, tokenizer, schedule, device: str,
                       n_batches: int = 20, t: float = 0.5):
    """Empirical residual-stream sigma per layer, measured under the noise the
    model actually sees. Expect it to increase with depth in a pre-LN network;
    if it does not, something is wrong with the checkpoint."""
    from ..backbone.hooks import ActivationCache
    from ..diffusion.objective import forward_mask

    n_layers = len(model.blocks) + 1
    cache = ActivationCache(mode="full", device="cpu")
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ids = batch["ids"].to(device)
        maskable = batch["maskable"].to(device)
        tv = torch.full((ids.shape[0],), float(t), device=device)
        x_t, _ = forward_mask(ids, maskable, tv, schedule, tokenizer.mask_id)
        model(x_t, tv, cache=cache)
    std = torch.stack([cache.stack(l).float().std() for l in range(n_layers)])
    cache.clear()
    return std


# --------------------------------------------------------- alpha calibration
def normalise_alpha(alpha_sigma: float, layer: int, stats) -> float:
    """Absolute coefficient for a given layer."""
    return float(alpha_sigma) * float(getattr(stats, "std", stats)[layer])


@dataclass
class PiecewiseAlpha:
    """Monotone interpolator from a target property value to a coefficient.

    Fitted on the observed dose-response curve. Used only when the response is
    monotone but materially non-linear; the non-linearity is still reported.
    """
    alphas: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        order = np.argsort(self.values)
        self.values = np.asarray(self.values, dtype=float)[order]
        self.alphas = np.asarray(self.alphas, dtype=float)[order]

    def __call__(self, target: float) -> float:
        return float(np.interp(target, self.values, self.alphas))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"alphas": self.alphas.tolist(),
                       "values": self.values.tolist()}, f, indent=2)

    @staticmethod
    def load(path: str) -> "PiecewiseAlpha":
        with open(path) as f:
            d = json.load(f)
        return PiecewiseAlpha(np.array(d["alphas"]), np.array(d["values"]))


def fit_piecewise_alpha(sweep_df, alpha_col="alpha", value_col="prop_mean"
                        ) -> PiecewiseAlpha:
    g = sweep_df.groupby(alpha_col)[value_col].mean().reset_index()
    return PiecewiseAlpha(g[alpha_col].to_numpy(), g[value_col].to_numpy())


def monotonicity(sweep_df, alpha_col="alpha", value_col="prop_mean") -> Dict:
    """Spearman rho between coefficient and realised property — the Tier 1
    criterion. Reported with the direction of the relationship so a sign flip
    is visible rather than absorbed into the magnitude."""
    from scipy.stats import spearmanr
    g = sweep_df.groupby(alpha_col)[value_col].mean().reset_index()
    r = spearmanr(g[alpha_col], g[value_col])
    return {"spearman": float(r.statistic), "p": float(r.pvalue),
            "n_points": int(len(g)),
            "passes_tier1": bool(abs(r.statistic) > 0.7)}