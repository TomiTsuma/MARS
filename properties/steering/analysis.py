"""
Representational analysis.

These functions produce the interpretability results the phase exists for.
Three of them are designed to be able to return an UNFAVOURABLE answer, and
that is deliberate:

  * `effective_dim` tests the approximate one-dimensionality claim rather than
    assuming it. A continuous property spreading across many components bounds
    what single-direction steering can achieve.

  * `offtarget_matrix` can reveal that several "different" property directions
    are the same latent factor (usually molecular size) under different names.

  * `fragment_frequency_shift` can reveal that steering works by substituting a
    small set of known substructures rather than through a distributed
    representation. That is a legitimate mechanism and an honest finding, but
    it is a different claim from the one the phase sets out to make.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


# ------------------------------------------------- effective dimensionality
def effective_dim(H_pos: torch.Tensor, H_neg: torch.Tensor,
                  layer: int, site: int = 0, n_components: int = 10) -> Dict:
    """PCA on PER-SAMPLE difference vectors.

    Pairs positive and negative samples (truncating to the shorter side) and
    performs PCA on the differences. The variance explained by PC1 is the
    direct test of the 'approximately one-dimensional' claim inherited from the
    text precedent.
    """
    from sklearn.decomposition import PCA

    n = min(H_pos.shape[0], H_neg.shape[0])
    D = (H_pos[:n, layer, site, :] - H_neg[:n, layer, site, :]).numpy()
    k = min(n_components, D.shape[0] - 1, D.shape[1])
    if k < 1:
        return {"variance_explained": [], "pc1": float("nan")}
    pca = PCA(n_components=k).fit(D)
    ve = pca.explained_variance_ratio_.tolist()
    return {"variance_explained": ve,
            "pc1": float(ve[0]),
            "n_for_90pct": int(np.searchsorted(np.cumsum(ve), 0.90) + 1),
            "approximately_1d": bool(ve[0] > 0.5)}


def direction_similarity(directions: Dict[str, torch.Tensor]) -> Dict:
    """Pairwise cosine similarity between named directions.

    High similarity between, say, LogP and MW directions is evidence that a
    single latent factor underlies both, and should be read alongside the
    off-target matrix rather than in isolation.
    """
    names = list(directions)
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        va = directions[a] / directions[a].norm().clamp(min=1e-8)
        for j, b in enumerate(names):
            vb = directions[b] / directions[b].norm().clamp(min=1e-8)
            M[i, j] = float((va * vb).sum())
    return {"names": names, "matrix": M}


def per_decile_directions(H: torch.Tensor, y: np.ndarray, layer: int,
                          site: int = 0, n_bins: int = 5) -> Dict:
    """Extract a direction in each property bin and compare them.

    Near-parallel directions across bins support the single-direction
    assumption. Systematically rotating directions indicate the property
    manifold is curved, which bounds single-direction steering and motivates
    the piecewise-alpha fallback. Reported either way.
    """
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    chunks = np.array_split(order, n_bins)
    centres, vs = [], []
    global_mean = H[:, layer, site, :].mean(0)
    for c in chunks:
        vs.append(H[c, layer, site, :].mean(0) - global_mean)
        centres.append(float(np.mean(y[c])))
    V = torch.stack(vs)
    V = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    C = (V @ V.T).numpy()
    off = C[np.triu_indices(len(vs), k=1)]
    return {"bin_centres": centres, "cosine_matrix": C,
            "mean_offdiag_cosine": float(np.mean(off)),
            "min_offdiag_cosine": float(np.min(off)),
            "curved_manifold": bool(np.min(off) < 0.7)}


# ------------------------------------------------------------- specificity
def offtarget_matrix(gen_df, properties: Sequence[str],
                     baseline_df, alpha_col: str = "alpha",
                     steered_col: str = "steered_property",
                     alpha_ref: Optional[float] = None) -> Dict:
    """Rows = steered property, columns = measured property.

    Normalised by each property's unconditional standard deviation so cells are
    comparable across properties with different natural scales.

    ALWAYS read alongside the heavy-atom column. Without it this matrix cannot
    distinguish a property axis from a molecular-size axis.
    """
    import pandas as pd

    base_mean = {p: float(np.nanmean(baseline_df[p])) for p in properties}
    base_std = {p: float(np.nanstd(baseline_df[p])) or 1.0 for p in properties}

    rows = []
    for steered, grp in gen_df.groupby(steered_col):
        if alpha_ref is not None:
            grp = grp[np.isclose(grp[alpha_col], alpha_ref)]
        if grp.empty:
            continue
        rec = {"steered": steered}
        for p in properties:
            if p not in grp:
                continue
            rec[p] = (float(np.nanmean(grp[p])) - base_mean[p]) / base_std[p]
        if "heavy" in grp:
            rec["heavy_shift"] = float(np.nanmean(grp["heavy"])) - base_mean.get(
                "heavy", float(np.nanmean(baseline_df.get("heavy", [0]))))
        rows.append(rec)

    M = pd.DataFrame(rows).set_index("steered")
    diag = [abs(M.loc[s, s]) for s in M.index if s in M.columns]
    offd = [abs(M.loc[s, c]) for s in M.index for c in properties
            if c in M.columns and c != s]
    return {"matrix": M,
            "diagonal_mean": float(np.mean(diag)) if diag else float("nan"),
            "offdiagonal_mean": float(np.mean(offd)) if offd else float("nan"),
            "diagonal_dominance": (float(np.mean(diag) / (np.mean(offd) + 1e-8))
                                   if diag and offd else float("nan"))}


def fragment_frequency_shift(gen_df, alpha_col: str = "alpha",
                             smiles_col: str = "smiles",
                             top_k: int = 25) -> Dict:
    """Which substructures become more or less common as the coefficient rises.

    Detects steering-by-fragment-substitution. If a handful of fragments
    account for most of the property shift, the mechanism is substitution
    rather than a distributed representation — reportable as a mechanistic
    finding, but it changes the interpretability claim substantially.
    """
    from collections import Counter
    from rdkit import Chem
    from rdkit.Chem import BRICS
    import pandas as pd

    def frags(smis):
        c = Counter()
        for s in smis:
            m = Chem.MolFromSmiles(s) if s else None
            if m is None:
                continue
            try:
                for f in BRICS.BRICSDecompose(m):
                    c[f] += 1
            except Exception:
                continue
        return c

    alphas = sorted(gen_df[alpha_col].unique())
    lo = frags(gen_df[gen_df[alpha_col] == alphas[0]][smiles_col])
    hi = frags(gen_df[gen_df[alpha_col] == alphas[-1]][smiles_col])
    n_lo, n_hi = max(1, sum(lo.values())), max(1, sum(hi.values()))

    keys = set(lo) | set(hi)
    rows = [{"fragment": k,
             "freq_low_alpha": lo.get(k, 0) / n_lo,
             "freq_high_alpha": hi.get(k, 0) / n_hi,
             "delta": hi.get(k, 0) / n_hi - lo.get(k, 0) / n_lo} for k in keys]
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)

    top_share = float(df.head(top_k)["delta"].clip(lower=0).sum())
    return {"table": df, "alpha_low": alphas[0], "alpha_high": alphas[-1],
            "top_k_positive_share": top_share,
            "substitution_dominated": bool(top_share > 0.5)}