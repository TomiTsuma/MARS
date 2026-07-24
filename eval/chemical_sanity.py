"""
Chemical sanity — the detectors that guard the SELFIES trap.

WHY THIS MODULE EXISTS
----------------------
Schiff et al. (2024) report severe validity collapse when absorbing-state
masked diffusion models are pushed with classifier-free or classifier-based
guidance: in the strongest settings fewer than 10% of generated sequences
remained parseable. The mechanism is the carry-over constraint — a token, once
unmasked, is frozen, so a token chosen off-distribution cannot be revised and
the model must complete a molecule around a mistake it cannot revisit.

They measured validity as "can RDKit parse this SMILES". THIS MODEL EMITS
SELFIES, where every token sequence decodes to a valid molecule by
construction. The same damage will therefore occur and will be INVISIBLE:
validity reads 99-100% at every coefficient while the underlying chemistry
degrades, because SELFIES silently repairs an incoherent token sequence into a
syntactically valid but chemically implausible molecule.

CONSEQUENCE: validity is not a quality metric in Phase 2 and must never be
reported alone. The functions below are what actually measure the damage, and
they must be in place BEFORE the alpha sweep is run, not added afterwards.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

try:
    from properties.oracles.rdkit_props import _sa as _sa_score
except Exception:  # pragma: no cover
    def _sa_score(m):
        return float("nan")


# ------------------------------------------------------- strict sanitisation
def strict_sanitize(smiles: str) -> bool:
    """Full RDKit sanitisation including valence and kekulisation.

    Catches valence violations and aromaticity failures that permissive
    parsing tolerates and that SELFIES decoding can silently produce.
    """
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        Chem.Kekulize(Chem.Mol(mol), clearAromaticFlags=True)
        return True
    except Exception:
        return False


# ------------------------------------------------------ ring plausibility
def ring_systems(smiles: str) -> Set[str]:
    """Canonical SMILES of each fused ring system in the molecule."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return set()
    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return set()

    # merge rings sharing atoms into fused systems
    systems: List[Set[int]] = []
    for ring in ri.AtomRings():
        s = set(ring)
        merged = [t for t in systems if t & s]
        for t in merged:
            s |= t
            systems.remove(t)
        systems.append(s)

    out = set()
    for s in systems:
        try:
            sub = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(s),
                                           canonical=True)
            if sub:
                out.add(sub)
        except Exception:
            continue
    return out


def build_training_ring_systems(train_smiles: Iterable[str],
                                verbose: bool = True) -> Set[str]:
    """Reference set of ring systems seen in training. Run once and cache —
    it is the denominator for the plausibility check."""
    ref: Set[str] = set()
    for i, s in enumerate(train_smiles):
        ref |= ring_systems(s)
        if verbose and (i + 1) % 200_000 == 0:
            print(f"    ring systems from {i+1:,} molecules: {len(ref):,}")
    return ref


def ring_plausibility(smiles: str, train_rings: Set[str]) -> bool:
    """True if every ring system in the molecule was seen in training.

    Novel ring systems are the classic signature of off-manifold generation.
    A modest rate is normal; a rate that climbs with the steering coefficient
    is the damage signal.
    """
    rs = ring_systems(smiles)
    if not rs:
        return True                      # acyclic molecules are not implausible
    return all(r in train_rings for r in rs)


# --------------------------------------------------------------- the report
def sanity_report(smiles_list: Sequence[Optional[str]],
                  train_rings: Optional[Set[str]] = None,
                  baseline_sa: Optional[float] = None) -> Dict:
    """The fidelity signal for Phase 2. Report this, not validity.

    Returns
      selfies_validity      -- expected ~1.0; included ONLY to show it is
                               uninformative, never as evidence of quality
      strict_pass_rate      -- full sanitisation survival
      ring_plausible_rate   -- fraction with only seen ring systems
      sa_mean / sa_shift    -- synthetic accessibility drift vs baseline
      chemical_sanity       -- strict AND ring-plausible; the headline number
    """
    smis = [s for s in smiles_list if s]
    n_total = len(smiles_list)
    if not smis:
        return {"n": n_total, "selfies_validity": 0.0, "strict_pass_rate": 0.0,
                "ring_plausible_rate": 0.0, "chemical_sanity": 0.0,
                "sa_mean": float("nan"), "sa_shift": float("nan")}

    parseable = [s for s in smis if Chem.MolFromSmiles(s) is not None]
    strict = [s for s in parseable if strict_sanitize(s)]

    if train_rings is not None:
        plaus = [s for s in strict if ring_plausibility(s, train_rings)]
    else:
        plaus = strict

    sas = []
    for s in strict[:5000]:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            v = _sa_score(m)
            if np.isfinite(v):
                sas.append(v)
    sa_mean = float(np.mean(sas)) if sas else float("nan")

    return {
        "n": n_total,
        "selfies_validity": len(parseable) / max(1, n_total),
        "strict_pass_rate": len(strict) / max(1, n_total),
        "ring_plausible_rate": len(plaus) / max(1, len(strict)),
        "chemical_sanity": len(plaus) / max(1, n_total),
        "sa_mean": sa_mean,
        "sa_shift": (sa_mean - baseline_sa) if (baseline_sa is not None
                                                and np.isfinite(sa_mean))
                    else float("nan"),
    }


def trap_check(reports_by_alpha: Dict[float, Dict],
               validity_tol: float = 0.02,
               sanity_drop: float = 0.15) -> Dict:
    """Detect the SELFIES trap explicitly: validity flat while sanity falls.

    If this fires, the study is in the regime where reporting validity alone
    would have concluded — incorrectly — that steering is harmless.
    """
    alphas = sorted(reports_by_alpha)
    val = np.array([reports_by_alpha[a]["selfies_validity"] for a in alphas])
    san = np.array([reports_by_alpha[a]["chemical_sanity"] for a in alphas])
    validity_flat = bool((val.max() - val.min()) < validity_tol)
    sanity_fell = bool((san.max() - san.min()) > sanity_drop)
    return {"alphas": alphas,
            "validity_range": float(val.max() - val.min()),
            "sanity_range": float(san.max() - san.min()),
            "validity_flat": validity_flat,
            "sanity_degraded": sanity_fell,
            "trap_detected": bool(validity_flat and sanity_fell),
            "message": ("SELFIES TRAP DETECTED: validity is flat while chemical "
                        "sanity degrades. Do not report validity as evidence of "
                        "quality." if (validity_flat and sanity_fell) else
                        "no divergence between validity and chemical sanity")}