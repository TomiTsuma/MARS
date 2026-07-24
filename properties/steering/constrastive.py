"""
Contrastive set construction.

THE CENTRAL POINT OF THIS MODULE
--------------------------------
Heavy-atom matching is the SAMPLING ALGORITHM, not a filter applied after the
fact. Naively taking the top and bottom property deciles of a corpus produces
two sets that differ in molecular size as much as they differ in the property,
because most molecular properties correlate with size. A direction extracted
from such sets encodes "big molecule minus small molecule", and every
downstream result — dose-response, localisation, specificity — is confounded.

The fix is to stratify by heavy-atom count first and take property deciles
*within each stratum*, so the two sides have matched size distributions by
construction rather than by luck.

GATE A is the check that this worked: the property distributions must differ
strongly, and the heavy-atom distributions must not differ at all.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class ContrastiveSet:
    """Indices into the annotated corpus frame, plus provenance."""
    property: str
    positive_idx: List[int]
    negative_idx: List[int]
    n_per_side: int
    n_strata: int
    seed: int
    corpus: str = ""
    split: str = ""
    match_stats: Dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "ContrastiveSet":
        with open(path) as f:
            return ContrastiveSet(**json.load(f))


# ------------------------------------------------------------------ binning
def atom_count_bins(heavy: np.ndarray, n_bins: int = 8) -> List[Tuple[int, int]]:
    """Quantile bins over heavy-atom count.

    Quantile rather than equal-width so every stratum holds a usable number of
    molecules; equal-width bins leave the tails almost empty on drug-like
    corpora and the matching silently degrades there.
    """
    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(heavy, qs).astype(int))
    if edges.size < 2:
        return [(int(heavy.min()), int(heavy.max()) + 1)]
    return [(int(edges[i]), int(edges[i + 1]) + (1 if i == len(edges) - 2 else 0))
            for i in range(len(edges) - 1)]


# ------------------------------------------------------------- construction
def match_atom_counts(df,
                      property: str,
                      n_per_side: int = 256,
                      n_strata: int = 8,
                      decile: float = 0.10,
                      seed: int = 0,
                      corpus: str = "",
                      split: str = "") -> ContrastiveSet:
    """Build D+ / D- with matched heavy-atom distributions.

    For each heavy-atom stratum, take the top and bottom `decile` fraction by
    the target property and draw an equal quota from each side. Because the
    quota per stratum is the same for both sides, the size distributions match
    by construction.
    """
    rng = np.random.default_rng(seed)
    if property not in df.columns:
        raise KeyError(f"'{property}' not in corpus frame")
    if "heavy" not in df.columns:
        raise KeyError("corpus frame is missing the 'heavy' confound control")

    heavy = np.asarray(df["heavy"], dtype=float)
    vals = np.asarray(df[property], dtype=float)
    finite = np.isfinite(vals) & np.isfinite(heavy)

    bins = atom_count_bins(heavy[finite], n_strata)
    quota = max(1, n_per_side // max(1, len(bins)))

    pos_idx: List[int] = []
    neg_idx: List[int] = []

    for lo, hi in bins:
        sel = np.where(finite & (heavy >= lo) & (heavy < hi))[0]
        if sel.size < 4:
            continue
        v = vals[sel]
        order = np.argsort(v)
        k_tail = max(1, int(np.ceil(decile * sel.size)))
        low_pool = sel[order[:k_tail]]
        high_pool = sel[order[-k_tail:]]

        k = min(quota, low_pool.size, high_pool.size)
        if k < 1:
            continue
        pos_idx.extend(rng.choice(high_pool, size=k, replace=False).tolist())
        neg_idx.extend(rng.choice(low_pool, size=k, replace=False).tolist())

    # top up from the global tails if strata under-delivered, preserving
    # the size balance by adding matched pairs only
    if len(pos_idx) < n_per_side:
        pos_idx, neg_idx = _topup(df, property, pos_idx, neg_idx,
                                  n_per_side, decile, rng, finite)

    cset = ContrastiveSet(
        property=property,
        positive_idx=[int(i) for i in pos_idx[:n_per_side]],
        negative_idx=[int(i) for i in neg_idx[:n_per_side]],
        n_per_side=min(n_per_side, len(pos_idx), len(neg_idx)),
        n_strata=len(bins), seed=seed, corpus=corpus, split=split,
    )
    cset.match_stats = _match_stats(df, cset, property)
    return cset


def _topup(df, property, pos_idx, neg_idx, n_per_side, decile, rng, finite):
    """Add matched pairs from the global tails, pairing on heavy-atom count so
    the size balance is preserved rather than degraded by the top-up."""
    heavy = np.asarray(df["heavy"], dtype=float)
    vals = np.asarray(df[property], dtype=float)
    used = set(pos_idx) | set(neg_idx)
    order = np.argsort(np.where(finite, vals, np.nan))
    order = order[np.isfinite(vals[order])]
    k_tail = max(1, int(np.ceil(decile * order.size)))
    low_pool = [i for i in order[:k_tail] if i not in used]
    high_pool = [i for i in order[-k_tail:] if i not in used]

    hi_by_size: Dict[int, List[int]] = {}
    for i in high_pool:
        hi_by_size.setdefault(int(heavy[i]), []).append(i)

    for j in low_pool:
        if len(pos_idx) >= n_per_side:
            break
        size = int(heavy[j])
        for delta in (0, 1, -1, 2, -2):
            bucket = hi_by_size.get(size + delta)
            if bucket:
                pos_idx.append(bucket.pop())
                neg_idx.append(j)
                break
    return pos_idx, neg_idx


def _match_stats(df, cset: ContrastiveSet, property: str) -> Dict:
    p = np.asarray(df[property], dtype=float)
    h = np.asarray(df["heavy"], dtype=float)
    pi, ni = cset.positive_idx, cset.negative_idx
    if not pi or not ni:
        return {"error": "empty contrastive set"}

    prop_ks = stats.ks_2samp(p[pi], p[ni])
    heavy_ks = stats.ks_2samp(h[pi], h[ni])
    return {
        "n_positive": len(pi), "n_negative": len(ni),
        "property_mean_pos": float(np.mean(p[pi])),
        "property_mean_neg": float(np.mean(p[ni])),
        "property_ks_stat": float(prop_ks.statistic),
        "property_ks_p": float(prop_ks.pvalue),
        "heavy_mean_pos": float(np.mean(h[pi])),
        "heavy_mean_neg": float(np.mean(h[ni])),
        "heavy_ks_stat": float(heavy_ks.statistic),
        "heavy_ks_p": float(heavy_ks.pvalue),
        "heavy_mean_abs_diff": float(abs(np.mean(h[pi]) - np.mean(h[ni]))),
    }


# ------------------------------------------------------------------- GATE A
def gate_a_check(cset: ContrastiveSet,
                 property_p_max: float = 1e-6,
                 heavy_p_min: float = 0.05,
                 heavy_mean_tol: float = 0.5,
                 verbose: bool = True) -> bool:
    """GATE A — the contrastive sets separate on the property and NOT on size.

    Failing this and proceeding means extracting a molecular-size direction and
    reporting it as a property direction. Every downstream number would be
    confounded and the confound would be invisible in the results.
    """
    s = cset.match_stats
    checks = {
        f"property separates (KS p < {property_p_max:g})":
            s.get("property_ks_p", 1.0) < property_p_max,
        f"heavy-atom counts do NOT separate (KS p > {heavy_p_min})":
            s.get("heavy_ks_p", 0.0) > heavy_p_min,
        f"heavy-atom mean difference < {heavy_mean_tol}":
            s.get("heavy_mean_abs_diff", 99.0) < heavy_mean_tol,
    }
    if verbose:
        print(f"GATE A  —  contrastive sets for '{cset.property}'")
        print(f"  n = {s.get('n_positive')} / {s.get('n_negative')}   "
              f"property mean  {s.get('property_mean_pos'):.3f}  vs  "
              f"{s.get('property_mean_neg'):.3f}")
        print(f"  heavy-atom mean  {s.get('heavy_mean_pos'):.2f}  vs  "
              f"{s.get('heavy_mean_neg'):.2f}   "
              f"(|Δ| = {s.get('heavy_mean_abs_diff'):.3f})")
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    if verbose:
        print(f"  -> {'PASS' if ok else 'FAIL — do not extract from these sets'}")
        if not ok and not checks[f"heavy-atom mean difference < {heavy_mean_tol}"]:
            print("     Increase n_strata, or widen `decile` so strata are better populated.")
    return ok


# ------------------------------------------------------- per-decile variant
def decile_bins(df, property: str, n_bins: int = 5,
                n_per_bin: int = 256, seed: int = 0) -> List[np.ndarray]:
    """Index sets for the per-decile curvature analysis (Section 3.5).

    Directions extracted at successive property levels are compared by cosine
    similarity: near-parallel supports the single-direction assumption,
    systematically rotating indicates a curved manifold and bounds what
    single-direction steering can achieve.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(df[property], dtype=float)
    ok = np.where(np.isfinite(v))[0]
    order = ok[np.argsort(v[ok])]
    chunks = np.array_split(order, n_bins)
    return [rng.choice(c, size=min(n_per_bin, c.size), replace=False)
            for c in chunks]