"""
Generative evaluation.

Calibrate expectations before reading any number from this module:

  * VALIDITY will be ~100% because SELFIES guarantees it. It carries almost no
    information about model quality. Do not let it reassure you.
  * NOVELTY rises when the model drifts off-distribution. It is never reported
    without FCD alongside.
  * The real Tier 0 signals are FCD, Frag/Scaf similarity, and uniqueness.
    A model producing valid, unique, novel garbage scores well on three of six.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from fcd_torch import FCD
from rdkit.Chem import QED, Crippen

RDLogger.DisableLog("rdApp.*")

def canonical(smiles: Optional[str]) -> Optional[str]:
    if smiles is None or smiles == "":
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def basic_metrics(
        generated: Sequence[Optional[str]],
        train_smiles: Sequence[str]
    ) -> Dict[str, float]:
    n = len(generated)
    canon = [canonical(s) for s in generated]
    valid = [s for s in canon if s is not None]

    uniq = set(valid)
    train_set = set(train_smiles)
    novel = uniq - train_set

    return {
        "n_generated": n,
        "validity": len(valid) / max(1, n),
        "uniqueness": len(uniq) / max(1, len(valid)),
        "novelty": len(novel) / max(1, len(uniq)),
        "unique@1k": len(set(valid[:1000])) / max(1, min(1000, len(valid))),
        "unique@10k": len(set(valid[:10000])) / max(1, min(10000, len(valid))),
    }

def morgan_fingerprints(
        smiles: Sequence[str], 
        radius: int = 2, 
        nbits: int = 2048
    ):
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits))
    return fps

def internal_diversity(
        smiles: Sequence[str], 
        sample: int = 5000,
        seed: int = 0
    ) -> float:
    """Mean pairwise Tanimoto distance. Detects steering-induced collapse
    toward a narrow chemotype (used heavily in Phase 2)."""
    rng = np.random.default_rng(seed)
    if len(smiles) > sample:
        smiles = [smiles[i] for i in rng.choice(len(smiles), sample, replace=False)]
    fps = morgan_fingerprints(smiles)
    if len(fps) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(sims)
        count += len(sims)
    return 1.0 - total / max(1, count)


def snn(
        generated: Sequence[str], 
        reference: Sequence[str],
        sample: int = 5000, 
        seed: int = 0
    ) -> float:
    """Mean nearest-neighbour Tanimoto similarity to the reference set."""
    rng = np.random.default_rng(seed)
    g = [generated[i] for i in rng.choice(len(generated),
         min(sample, len(generated)), replace=False)]
    r = [reference[i] for i in rng.choice(len(reference),
         min(sample, len(reference)), replace=False)]
    gf, rf = morgan_fingerprints(g), morgan_fingerprints(r)
    if not gf or not rf:
        return 0.0
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(f, rf)) for f in gf]))


def scaffold_similarity(
        generated: Sequence[str],
        reference: Sequence[str]
    ) -> float:
    """Cosine similarity of Murcko scaffold frequency vectors."""
    def counts(sm):
        d = {}
        for s in sm:
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
            except Exception:
                continue
            d[sc] = d.get(sc, 0) + 1
        return d
    a, b = counts(generated), counts(reference)
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    if va.sum() == 0 or vb.sum() == 0:
        return 0.0
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

def fcd(generated: Sequence[str], reference: Sequence[str]) -> Optional[float]:
    """Frechet ChemNet Distance. Requires `fcd_torch`; the primary
    distributional-fidelity measure throughout Phases 1 and 2."""

    return float(FCD(device="cuda", n_jobs=4)(gen=list(generated), ref=list(reference)))


def property_summary(smiles: Sequence[str]) -> Dict[str, float]:
    """Distributional sanity check on the chemistry itself."""
    
    rows = []
    for s in smiles[:5000]:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        rows.append((Descriptors.MolWt(m), Crippen.MolLogP(m),
                     QED.qed(m), m.GetNumHeavyAtoms()))
    if not rows:
        return {}
    a = np.array(rows)
    names = ["mw", "logp", "qed", "heavy_atoms"]
    return {f"{n}_{stat}": float(f(a[:, i]))
            for i, n in enumerate(names)
            for stat, f in (("mean", np.mean), ("std", np.std))}

def evaluate(generated: Sequence[Optional[str]],
             train_smiles: Sequence[str],
             test_smiles: Sequence[str],
             with_fcd: bool = True) -> Dict[str, float]:
    out = basic_metrics(generated, train_smiles)
    valid = [canonical(s) for s in generated]
    valid = [s for s in valid if s is not None]
    if not valid:
        return out
    out["int_div"] = internal_diversity(valid)
    out["snn_test"] = snn(valid, test_smiles)
    out["scaf_sim_test"] = scaffold_similarity(valid, test_smiles)
    if with_fcd:
        f = fcd(valid, list(test_smiles))
        if f is not None:
            out["fcd_test"] = f
    out.update(property_summary(valid))
    return out


# ------------------------------------------------------------------ TIER 0 GATE
TIER0 = {"validity": 0.99, "uniqueness": 0.95, "fcd_test_max": 1.5}


def tier0_gate(metrics: Dict[str, float]) -> bool:
    """GATE 4. No Phase 2 steering work begins until this passes.

    The FCD ceiling is indicative: check current published MOSES baselines for
    your split rather than trusting a hard-coded constant. The purpose of the
    gate is that findings about internal representations are attributable to
    representation structure and not to undertraining.
    """
    checks = {
        "validity >= 0.99": metrics.get("validity", 0) >= TIER0["validity"],
        "uniqueness >= 0.95": metrics.get("uniqueness", 0) >= TIER0["uniqueness"],
        f"fcd <= {TIER0['fcd_test_max']}":
            metrics.get("fcd_test", 1e9) <= TIER0["fcd_test_max"],
    }
    print("TIER 0 GATE")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"  -> {'PASS: proceed to Phase 2' if ok else 'FAIL: do not proceed'}")
    return ok


