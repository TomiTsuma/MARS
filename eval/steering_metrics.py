"""
Steering efficacy metrics.

Phase 1 metrics measure distribution quality; none of them measure control.
These do. The Tier 1 criterion (Spearman rho > 0.7) and the headline artefact
(the Pareto frontier) both live here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def property_shift(gen_values, baseline_values, n_boot: int = 1000,
                   seed: int = 0) -> Dict:
    """Mean shift with a bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    g = np.asarray(gen_values, dtype=float); g = g[np.isfinite(g)]
    b = np.asarray(baseline_values, dtype=float); b = b[np.isfinite(b)]
    if g.size == 0 or b.size == 0:
        return {"shift": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "cohens_d": float("nan")}
    shift = float(g.mean() - b.mean())
    boots = [rng.choice(g, g.size, True).mean() - rng.choice(b, b.size, True).mean()
             for _ in range(n_boot)]
    pooled = np.sqrt((g.var() + b.var()) / 2) or 1.0
    return {"shift": shift,
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "cohens_d": float(shift / pooled),
            "n_gen": int(g.size), "n_base": int(b.size)}


def spearman_dose(alphas, values) -> Dict:
    """rho(alpha, realised property) — the Tier 1 monotonicity criterion.

    Monotonicity matters more than linearity here. Saturation is acceptable and
    expected; non-monotonicity indicates the property manifold is curved enough
    to defeat single-direction steering.
    """
    from scipy.stats import spearmanr, pearsonr
    a = np.asarray(alphas, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(a) & np.isfinite(v)
    if ok.sum() < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "passes_tier1": False}
    s = spearmanr(a[ok], v[ok])
    p = pearsonr(a[ok], v[ok])
    return {"spearman": float(s.statistic), "spearman_p": float(s.pvalue),
            "pearson": float(p.statistic),
            "linearity_gap": float(abs(s.statistic) - abs(p.statistic)),
            "passes_tier1": bool(abs(s.statistic) > 0.7)}


def success_rate(values, target: float, tol: float) -> float:
    """Fraction within tolerance of the target — often more decision-relevant
    than the mean error."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(v - target) <= tol)) if v.size else float("nan")


def atom_count_stratified_shift(gen_df, baseline_df, prop: str,
                                bins=(0, 15, 20, 25, 30, 100)) -> Dict:
    """Property shift within heavy-atom strata.

    THE confound check. If the shift vanishes once molecular size is held
    fixed, the direction encodes size rather than the property and the headline
    result is not what it appears to be.
    """
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = gen_df[(gen_df.heavy >= lo) & (gen_df.heavy < hi)][prop]
        b = baseline_df[(baseline_df.heavy >= lo) & (baseline_df.heavy < hi)][prop]
        if len(g) < 20 or len(b) < 20:
            continue
        out[f"{lo}-{hi}"] = property_shift(g, b, n_boot=200)
    shifts = [v["shift"] for v in out.values()]
    return {"strata": out,
            "mean_within_stratum_shift": float(np.mean(shifts)) if shifts else float("nan"),
            "n_strata": len(out)}


def pareto_frontier(df, x: str = "delta_property", y: str = "fcd",
                    maximise_x: bool = True, minimise_y: bool = True):
    """Non-dominated points. THE headline artefact of Phase 2.

    A method is useful if its frontier lies below and to the right of the
    baselines. If classifier guidance dominates at every point, H5 is falsified.
    """
    import pandas as pd
    d = df.dropna(subset=[x, y]).copy()
    d["_x"] = d[x].abs() if maximise_x else -d[x].abs()
    d["_y"] = d[y] if minimise_y else -d[y]
    d = d.sort_values("_x", ascending=False)
    front, best_y = [], np.inf
    for _, r in d.iterrows():
        if r["_y"] < best_y:
            front.append(r)
            best_y = r["_y"]
    out = pd.DataFrame(front).drop(columns=["_x", "_y"], errors="ignore")
    return out.sort_values(x)


def summarise_condition(gen_df, baseline_df, prop: str,
                        sanity: Optional[Dict] = None) -> Dict:
    """One row of the results table for a single steering condition."""
    rec = {"property": prop}
    rec.update({f"shift_{k}": v for k, v in
                property_shift(gen_df[prop], baseline_df[prop]).items()})
    if "heavy" in gen_df and "heavy" in baseline_df:
        rec["heavy_shift"] = float(np.nanmean(gen_df.heavy)
                                   - np.nanmean(baseline_df.heavy))
    if sanity:
        rec.update({f"sanity_{k}": v for k, v in sanity.items()
                    if isinstance(v, (int, float))})
    return rec