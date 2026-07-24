"""
Activation collection and direction extraction.

This module has two jobs, and the SECOND one is the make-or-break check of
Phase 2:

  1. Collect residual-stream activations over contrastive sets and turn them
     into steering directions (difference-of-means, ridge probe).

  2. `probe_site_designs` — determine, using FORWARD PASSES ONLY and without
     generating a single molecule, whether each candidate extraction site
     actually carries property information.

Why (2) matters so much
-----------------------
The structural prefix tokens are identical in every sequence. At layer 0 their
activations are literally the same vector for every molecule (variance zero);
they acquire molecule-specific content only as bidirectional attention mixes
the sequence in. Nothing in the Phase 1 training objective rewarded putting
property information there. So the primary extraction-site design (S1) may
simply be inert — and if it is, that must be discovered before any generation
budget is spent, not after.

Reading the probe R^2 curve across depth:

    < 0.10   S1 is dead. Switch primary site to mean-pooled. Report it.
    0.10-0.40  Weak but usable. Expect small effect sizes; plan accordingly.
    > 0.50   S1 is strong. Proceed with confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model.hooks import ActivationCache
from model.objective import forward_mask


# ============================================================ site reducers
class SiteReducer:
    """Reduces a (B, L, D) residual-stream tensor to (B, n_sites, D).

    Reduction happens per batch inside the collection loop so the full
    (n_mol, n_layers, L, D) tensor is never materialised. For 512 molecules at
    L=128 that tensor would be ~2.2 GB; the prefix reduction is ~150 MB.
    """
    name = "base"
    n_sites = 1

    def __call__(self, h: torch.Tensor, ids: torch.Tensor,
                 n_struct: int, pad_id: int, mask_id: int) -> torch.Tensor:
        raise NotImplementedError


class PrefixReducer(SiteReducer):
    """S1 — the structural prefix (BOS + P0..P7). Position-invariant by
    construction; the direct analogue of chat-template extraction."""
    name = "prefix"

    def __init__(self, n_struct: int = 9):
        self.n_sites = n_struct

    def __call__(self, h, ids, n_struct, pad_id, mask_id):
        return h[:, :self.n_sites, :]


class PooledReducer(SiteReducer):
    """S2 — mean over body (non-prefix, non-pad) positions. Robust; discards
    the positional localisation signal the study exists to measure, which is
    exactly why it is the baseline rather than the primary."""
    name = "pooled"
    n_sites = 1

    def __call__(self, h, ids, n_struct, pad_id, mask_id):
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device)
        m = (pos >= n_struct).unsqueeze(0) & ids.ne(pad_id)
        m = m.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)


class MaskReducer(SiteReducer):
    """S3 — mean over positions currently holding [MASK]. Diffusion-native,
    with no autoregressive analogue. Requires t > 0 collection; at t = 0 there
    are no masks and this reducer returns zeros."""
    name = "masked"
    n_sites = 1

    def __call__(self, h, ids, n_struct, pad_id, mask_id):
        m = ids.eq(mask_id).unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)


REDUCERS = {"prefix": PrefixReducer, "pooled": PooledReducer, "masked": MaskReducer}


# ======================================================= activation capture
@torch.no_grad()
def collect_activations(model,
                        ids: torch.Tensor,
                        tokenizer,
                        schedule=None,
                        reducer: Optional[SiteReducer] = None,
                        layers: Optional[Sequence[int]] = None,
                        t: float = 0.0,
                        n_realizations: int = 1,
                        batch_size: int = 64,
                        device: str = "cuda",
                        seed: int = 0) -> torch.Tensor:
    """Collect residual-stream activations.

    Returns (n_mol, n_layers, n_sites, d_model), float32 on CPU.

    t = 0.0  ->  clean molecules, no masking (extraction option A, default).
                 Cleanest estimator, lowest variance, direct analogue of the
                 prompt-only extraction used in the text precedent.

    t > 0.0  ->  forward-masked molecules averaged over `n_realizations`
                 (extraction option B). Distribution-matched to the conditions
                 under which steering is actually applied, but noisier. Running
                 both is what measures the clean-extract / noisy-apply mismatch.
    """
    model.eval()
    n_layers_total = len(model.blocks) + 1          # blocks + final residual
    layers = list(layers) if layers is not None else list(range(n_layers_total))
    reducer = reducer or PrefixReducer(tokenizer.n_struct)

    gen = torch.Generator(device=device).manual_seed(seed)
    out_chunks: List[torch.Tensor] = []

    for start in range(0, ids.shape[0], batch_size):
        batch = ids[start:start + batch_size].to(device)
        acc = None
        reps = max(1, n_realizations if t > 0 else 1)

        for _ in range(reps):
            if t > 0.0:
                from datasets.dataset import make_masks
                maskable, _ = make_masks(batch, tokenizer.pad_id, tokenizer.n_struct)
                tv = torch.full((batch.shape[0],), float(t), device=device)
                x_t, _ = forward_mask(batch, maskable, tv, schedule,
                                      tokenizer.mask_id, generator=gen)
            else:
                x_t = batch
                tv = torch.full((batch.shape[0],), 1e-3, device=device)

            cache = ActivationCache(mode="full", layers=layers, device="cpu")
            model(x_t, tv, cache=cache)

            # (B, n_layers, n_sites, D)
            per_layer = []
            for l in layers:
                h = cache.stack(l).to(device)            # (B, L, D)
                per_layer.append(reducer(h, x_t, tokenizer.n_struct,
                                         tokenizer.pad_id, tokenizer.mask_id))
            stacked = torch.stack(per_layer, dim=1).float().cpu()
            acc = stacked if acc is None else acc + stacked
            cache.clear()

        out_chunks.append(acc / reps)

    return torch.cat(out_chunks, dim=0)


# ====================================================== direction estimators
def diff_of_means(H_pos: torch.Tensor, H_neg: torch.Tensor) -> torch.Tensor:
    """Difference-of-means direction, L2-normalised.

    H_*: (n_mol, n_layers, n_sites, d)  ->  (n_layers, n_sites, d)

    Natural for binary endpoints. For continuous properties it discards the
    central mass of the distribution through decile binarisation, which is why
    the ridge probe below is run alongside it as a methodological comparison
    rather than as hyperparameter selection.
    """
    v = H_pos.mean(0) - H_neg.mean(0)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def ridge_probe(H: torch.Tensor, y: np.ndarray,
                alpha: float = 1.0) -> torch.Tensor:
    """Ridge-regression direction: regress activations onto the CONTINUOUS
    property value and take the normalised weight vector.

    H: (n_mol, n_layers, n_sites, d)  ->  (n_layers, n_sites, d)
    Uses the full property distribution rather than only the tails.
    """
    from sklearn.linear_model import Ridge

    n, nl, ns, d = H.shape
    out = torch.zeros(nl, ns, d)
    X_all = H.numpy()
    y = np.asarray(y, dtype=float)
    for l in range(nl):
        for s in range(ns):
            X = X_all[:, l, s, :]
            if np.allclose(X.std(0).max(), 0.0):     # constant site (layer 0 prefix)
                continue
            w = Ridge(alpha=alpha).fit(X, y).coef_
            nrm = np.linalg.norm(w)
            if nrm > 1e-8:
                out[l, s] = torch.from_numpy(w / nrm).float()
    return out


def bootstrap_directions(H_pos: torch.Tensor, H_neg: torch.Tensor,
                         n_boot: int = 100, seed: int = 0
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direction stability under resampling of the contrastive sets.

    Returns (mean_direction, cosine_similarity_to_full_estimate_std).
    A direction whose bootstrap cosine spread is wide is not a stable estimate
    and should not be reported as a finding.
    """
    rng = np.random.default_rng(seed)
    full = diff_of_means(H_pos, H_neg)
    cos = []
    for _ in range(n_boot):
        pi = rng.integers(0, H_pos.shape[0], H_pos.shape[0])
        ni = rng.integers(0, H_neg.shape[0], H_neg.shape[0])
        v = diff_of_means(H_pos[pi], H_neg[ni])
        cos.append((v * full).sum(-1).numpy())        # (n_layers, n_sites)
    cos = np.stack(cos, 0)
    return full, torch.from_numpy(cos.std(0)).float()


# ================================================= THE STAGE B PROBE
def probe_site_designs(H_by_site: Dict[str, torch.Tensor],
                       y: np.ndarray,
                       test_frac: float = 0.3,
                       alpha: float = 1.0,
                       seed: int = 0) -> Dict[str, Dict]:
    """Ridge-regress activations onto the property, per layer, per site design.

    This is the decisive Phase 2 check and it costs a few hundred forward
    passes. It answers experiment E1's main question — which extraction site
    carries property information — WITHOUT generating a single molecule.

    H_by_site: {"prefix": (n,nl,ns,d), "pooled": (n,nl,1,d), ...}
    Returns   {site: {"r2_per_layer": [...], "best_layer": int, "best_r2": float}}

    Note: at layer 0 the prefix positions are identical across molecules
    (variance zero), so R^2 there will be ~0 by construction. That is a
    CORRECTNESS CHECK, not a failure: if layer 0 shows high R^2 something is
    wrong with the collection code.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    y = np.asarray(y, dtype=float)
    results: Dict[str, Dict] = {}

    for site, H in H_by_site.items():
        n, nl, ns, d = H.shape
        X_all = H.reshape(n, nl, ns * d).numpy()
        r2s: List[float] = []
        for l in range(nl):
            X = X_all[:, l, :]
            if X.std(0).max() < 1e-8:                # constant → no signal
                r2s.append(0.0)
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_frac, random_state=seed)
            model = Ridge(alpha=alpha).fit(Xtr, ytr)
            r2s.append(float(model.score(Xte, yte)))   # held-out R^2
        best = int(np.argmax(r2s))
        results[site] = {"r2_per_layer": r2s,
                         "best_layer": best,
                         "best_r2": float(r2s[best])}
    return results


def interpret_probe(results: Dict[str, Dict], verbose: bool = True) -> Dict:
    """GATE B — decide the primary extraction site from the probe."""
    verdicts = {}
    for site, r in results.items():
        r2 = r["best_r2"]
        if r2 < 0.10:
            v = "INERT — carries no usable property signal"
        elif r2 < 0.40:
            v = "WEAK — usable, expect small effect sizes"
        elif r2 < 0.50:
            v = "MODERATE"
        else:
            v = "STRONG — proceed with confidence"
        verdicts[site] = {"best_r2": r2, "best_layer": r["best_layer"], "verdict": v}

    ranked = sorted(verdicts.items(), key=lambda kv: -kv[1]["best_r2"])
    primary = ranked[0][0]

    if verbose:
        print("GATE B  —  extraction-site linear probe (held-out R^2)")
        for site, v in ranked:
            print(f"  {site:>8}:  R^2 = {v['best_r2']:.3f}  "
                  f"at layer {v['best_layer']:>2}   {v['verdict']}")
        print(f"  -> primary site: {primary}")
        if verdicts.get("prefix", {}).get("best_r2", 0) < 0.10:
            print("     WARNING: the structural prefix is inert. Switch the primary")
            print("     extraction site to 'pooled' and report this as a finding —")
            print("     it means property information is distributed, not localised.")
    return {"verdicts": verdicts, "primary_site": primary,
            "prefix_inert": verdicts.get("prefix", {}).get("best_r2", 0) < 0.10}


def projection_check(H_heldout: torch.Tensor, v: torch.Tensor,
                     y: np.ndarray, layer: int, site: int = 0) -> float:
    """Spearman correlation between projection onto the direction and the
    property, on HELD-OUT molecules.

    The cheapest sanity check available, and the one everyone skips. If this is
    near zero the direction is noise and no amount of coefficient tuning will
    rescue it — go back to extraction, not to alpha. Stored on every
    DirectionArtifact for exactly this reason.
    """
    from scipy.stats import spearmanr
    proj = (H_heldout[:, layer, site, :] @ v).numpy()
    return float(spearmanr(proj, np.asarray(y, dtype=float)).statistic)