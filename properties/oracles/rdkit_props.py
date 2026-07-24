"""
Exact, deterministic property oracles (RDKit).

Phase 2 validates the steering method against these before touching any
predicted endpoint. The reason is stated in the specification: RDKit oracles
have ZERO oracle error, so any observed failure is unambiguously a failure of
the steering method rather than an artefact of label noise or of a surrogate
model evaluated off-distribution.

`heavy` and `mw` are in the default property set deliberately and are NOT
optional. They are the standing confound check: most molecular properties
correlate with molecular size, and a steering direction that merely increases
size will improve MAE on several properties at once and look like control.
Every generated molecule must carry them.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# ---- synthetic accessibility lives in RDKit Contrib, not the main API -------
_SA_OK = False
try:
    from rdkit.Chem import RDConfig
    _sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if _sa_path not in sys.path:
        sys.path.append(_sa_path)
    import sascorer  # type: ignore
    _SA_OK = True
except Exception:  # pragma: no cover
    sascorer = None


def _sa(mol) -> float:
    if not _SA_OK:
        return float("nan")
    return float(sascorer.calculateScore(mol))


# ---------------------------------------------------------------- registry
PROPERTIES = {
    # steerable targets
    "logp":  lambda m: float(Crippen.MolLogP(m)),
    "qed":   lambda m: float(QED.qed(m)),
    "tpsa":  lambda m: float(Descriptors.TPSA(m)),
    "sa":    _sa,
    "rings": lambda m: int(rdMolDescriptors.CalcNumAromaticRings(m)),
    # ---- confound controls: always computed, never optional ----
    "heavy": lambda m: int(m.GetNumHeavyAtoms()),
    "mw":    lambda m: float(Descriptors.MolWt(m)),
}

CONFOUND_CONTROLS = ("heavy", "mw")
DEFAULT_PROPS = tuple(PROPERTIES.keys())

# Properties on which steering is validated first (exact oracles only).
EXACT_TARGETS = ("logp", "qed", "tpsa", "sa", "rings")


def canonical(smiles: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def compute(smiles: str, props: Sequence[str] = DEFAULT_PROPS) -> Optional[Dict]:
    """Compute properties for one molecule. Returns None if unparseable.

    Confound controls are force-included regardless of `props`.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    wanted = list(dict.fromkeys(list(props) + list(CONFOUND_CONTROLS)))
    out: Dict[str, object] = {"smiles": Chem.MolToSmiles(mol, canonical=True)}
    for p in wanted:
        fn = PROPERTIES.get(p)
        if fn is None:
            raise KeyError(f"unknown property '{p}'; known: {sorted(PROPERTIES)}")
        try:
            out[p] = fn(mol)
        except Exception:
            out[p] = float("nan")
    return out


def _worker(smi: str):
    return compute(smi)


def annotate_corpus(smiles_list: Iterable[str],
                    out_path: Optional[str] = None,
                    props: Sequence[str] = DEFAULT_PROPS,
                    n_jobs: int = 8,
                    verbose: bool = True):
    """Annotate a corpus and optionally cache it as parquet.

    Run this ONCE per corpus. Property computation over ~2M molecules is
    minutes with multiprocessing and hours without; downstream sweeps read
    from the cache rather than recomputing.
    """
    import pandas as pd

    smiles_list = list(smiles_list)
    rows: List[Optional[Dict]] = []

    if n_jobs and n_jobs > 1 and len(smiles_list) > 512:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            for i, r in enumerate(pool.imap(_worker, smiles_list, chunksize=256)):
                rows.append(r)
                if verbose and (i + 1) % 100_000 == 0:
                    print(f"    annotated {i+1:,}")
    else:
        for smi in smiles_list:
            rows.append(compute(smi, props))

    kept = [r for r in rows if r is not None]
    df = pd.DataFrame(kept)
    if verbose:
        print(f"    annotated {len(df):,} / {len(smiles_list):,} "
              f"({len(smiles_list)-len(df):,} unparseable)")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_parquet(out_path, index=False)
        if verbose:
            print(f"    wrote {out_path}")
    return df


def summarise(df, props: Sequence[str] = EXACT_TARGETS) -> Dict[str, Dict]:
    """Distribution summary — used to sanity-check a corpus before use."""
    out = {}
    for p in list(props) + list(CONFOUND_CONTROLS):
        if p not in df.columns:
            continue
        v = np.asarray(df[p], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[p] = {"mean": float(v.mean()), "std": float(v.std()),
                  "min": float(v.min()), "max": float(v.max()),
                  "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90))}
    return out