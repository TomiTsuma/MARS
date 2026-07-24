"""
Candidate sweep and operating-point selection.

Extraction yields one candidate direction per (layer, position) pair. This
module evaluates them all and picks one.

COST NOTE — the single biggest lever on total Phase 2 wall-clock.
Sweeping uses a REDUCED generation budget (default 500 molecules, N=32, one
seed). The full reporting protocol (10,000 molecules, N=128, three seeds) is
roughly 15x more expensive per condition and is reserved for final numbers.
Sweeping at reporting budget is the most common way this phase runs long.

The full sweep is retained, not just the argmax: the layer-by-position
sensitivity map IS the interpretability result. A tight hotspot means property
information is localised; a diffuse field means it is distributed. Both are
findings.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from model.sampler import sample_to_smiles
from properties.steering.intervene import AdditiveSteer
from properties.steering.positions import from_spec as position_from_spec
from properties.steering.schedules import from_spec as schedule_from_spec


def _fcd_proxy(smiles: List[str], ref_fps) -> float:
    """Cheap stand-in for FCD during sweeps.

    Mean nearest-neighbour Tanimoto DISTANCE to a reference sample: cheap,
    correlates with distributional departure, and needs no ChemNet forward
    pass. Final numbers still use real FCD — this is for ranking only.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
    if not fps or not ref_fps:
        return float("nan")
    sims = [max(DataStructs.BulkTanimotoSimilarity(f, ref_fps)) for f in fps]
    return float(1.0 - np.mean(sims))


def reference_fingerprints(smiles: Sequence[str], n: int = 2000, seed: int = 0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(smiles), size=min(n, len(smiles)), replace=False)
    out = []
    for i in sel:
        m = Chem.MolFromSmiles(smiles[i])
        if m is not None:
            out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
    return out


def sweep_candidates(model, tokenizer, schedule, directions: torch.Tensor,
                     property_fn: Callable[[List[str]], np.ndarray],
                     stats, ref_fps,
                     baseline_prop: float,
                     alpha_ref: float = 2.0,
                     n_mols: int = 500,
                     n_steps: int = 32,
                     layers: Optional[Sequence[int]] = None,
                     positions: Optional[Sequence[int]] = None,
                     position_spec: str = "all",
                     schedule_spec: str = "all",
                     device: str = "cuda",
                     seed: int = 0,
                     verbose: bool = True):
    """Evaluate every (layer, position) candidate at a fixed reference alpha.

    `directions` is (n_layers, n_sites, d) as returned by the extractors.
    Returns a DataFrame with one row per candidate.
    """
    import pandas as pd

    n_layers, n_sites, _ = directions.shape
    layers = list(layers) if layers is not None else list(range(n_layers))
    positions = list(positions) if positions is not None else list(range(n_sites))

    rows = []
    total = len(layers) * len(positions)
    for c, L in enumerate(layers):
        for P in positions:
            v = directions[L, P]
            if float(v.norm()) < 1e-6:      # e.g. constant layer-0 prefix
                rows.append({"layer": L, "position": P, "delta_property": 0.0,
                             "prop_mean": float("nan"), "fcd_proxy": float("nan"),
                             "heavy_shift": 0.0, "n_valid": 0, "skipped": True})
                continue

            steer = AdditiveSteer(
                v, alpha_ref, stats=stats, layers=None,
                schedule=schedule_from_spec(schedule_spec),
                positions=position_from_spec(position_spec),
                n_struct=tokenizer.n_struct, pad_id=tokenizer.pad_id,
                mask_id=tokenizer.mask_id)

            torch.manual_seed(seed)
            gen = sample_to_smiles(model, schedule, tokenizer, n_mols,
                                   n_steps=n_steps, device=device, steer=steer)
            valid = [s for s in gen if s]
            if not valid:
                rows.append({"layer": L, "position": P, "delta_property": 0.0,
                             "prop_mean": float("nan"), "fcd_proxy": float("nan"),
                             "heavy_shift": 0.0, "n_valid": 0, "skipped": False})
                continue

            props = property_fn(valid)
            pm = float(np.nanmean(props["target"]))
            rows.append({
                "layer": L, "position": P,
                "prop_mean": pm,
                "delta_property": pm - baseline_prop,
                "fcd_proxy": _fcd_proxy(valid, ref_fps),
                "heavy_shift": float(np.nanmean(props["heavy"]) - props["heavy_base"]),
                "n_valid": len(valid), "skipped": False,
            })
            if verbose and (c * len(positions) + positions.index(P)) % 10 == 0:
                done = c * len(positions) + positions.index(P) + 1
                print(f"    [{done}/{total}] layer {L} pos {P}: "
                      f"Δ={rows[-1]['delta_property']:+.3f} "
                      f"fcd~{rows[-1]['fcd_proxy']:.3f}")
    return pd.DataFrame(rows)


def select_operating_point(df, max_fcd_proxy: Optional[float] = None,
                           max_heavy_shift: float = 1.0,
                           verbose: bool = True):
    """argmax |delta property| subject to fidelity and size constraints.

    The size constraint is not optional: a candidate that achieves a large
    property shift by making molecules bigger is the confound this programme
    exists to control, and it must not be selected as the operating point.
    """
    ok = df[~df["skipped"].astype(bool) & (df["n_valid"] > 0)].copy()
    if max_fcd_proxy is not None:
        ok = ok[ok["fcd_proxy"] <= max_fcd_proxy]
    ok = ok[ok["heavy_shift"].abs() <= max_heavy_shift]
    if ok.empty:
        if verbose:
            print("  no candidate satisfies the constraints — "
                  "relax max_fcd_proxy or re-examine extraction")
        return None
    best = ok.loc[ok["delta_property"].abs().idxmax()]
    if verbose:
        print(f"  selected layer {int(best.layer)} position {int(best.position)}: "
              f"Δ={best.delta_property:+.3f}  fcd~{best.fcd_proxy:.3f}  "
              f"heavy_shift={best.heavy_shift:+.2f}")
    return {"layer": int(best.layer), "position": int(best.position),
            "delta_property": float(best.delta_property),
            "fcd_proxy": float(best.fcd_proxy),
            "heavy_shift": float(best.heavy_shift)}


def heatmap_data(df, n_layers: int, n_positions: int,
                 value: str = "delta_property") -> np.ndarray:
    """(n_layers, n_positions) grid for Figure A."""
    Z = np.full((n_layers, n_positions), np.nan)
    for _, r in df.iterrows():
        Z[int(r["layer"]), int(r["position"])] = r[value]
    return Z


def plot_heatmap(Z, out_path: str, title: str = "",
                 selected=None, position_labels=None):
    """Figure A. Read the structure, not just the argmax."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "tp", ["#FFFFFF", "#DEEBF7", "#2E75B6", "#1F3864"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(np.abs(Z).T, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xlabel("extraction layer")
    ax.set_ylabel("extraction position")
    if position_labels:
        ax.set_yticks(range(len(position_labels)))
        ax.set_yticklabels(position_labels)
    ax.set_title(title or "Layer × position steering sensitivity",
                 color="#1F3864", fontweight="bold")
    if selected:
        ax.plot(selected["layer"], selected["position"], "s", ms=13,
                mfc="none", mec="#C55A11", mew=2.2)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015).set_label("|Δ property|")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path