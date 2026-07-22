from __future__ import annotations


import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
import pandas as pd


import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from datasets.tokenizer import SelfiesTokenizer
RDLogger.DisableLog("rdApp.*")

ALLOWED_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"}


# ---------------------------------------------------------------- standardise
def standardise(smiles: str,
                max_heavy: int = 60,
                allowed: Optional[set] = None) -> Optional[str]:
    """Canonical, desalted, neutral-ish SMILES, or None if rejected."""
    allowed = allowed or ALLOWED_ELEMENTS
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Largest fragment (drops salts and solvent co-crystals)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    if mol.GetNumHeavyAtoms() > max_heavy or mol.GetNumHeavyAtoms() < 4:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            return None
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return None                      # keep the neutral subset for Phase 1

    return Chem.MolToSmiles(mol, canonical=True)


# ---------------------------------------------------------------- round-trip
def to_selfies_checked(smiles: str) -> Optional[str]:
    """SMILES -> SELFIES, verified by decoding back to the same canonical SMILES."""
    try:
        s = sf.encoder(smiles)
        back = sf.decoder(s)
        m = Chem.MolFromSmiles(back)
        if m is None:
            return None
        if Chem.MolToSmiles(m, canonical=True) != smiles:
            return None
        return s
    except Exception:
        return None



def build_corpus(
        smiles_list: Iterable[str],
        max_heavy: int = 60,
        report_every: int = 100_000
    ) -> Tuple[List[str], List[str], Dict]:
    """Returns (canonical_smiles, selfies, stats)."""
    kept_smi, kept_sf = [], []
    stats = defaultdict(int)
    for i, smi in enumerate(smiles_list):
        stats["seen"] += 1
        can = standardise(smi, max_heavy=max_heavy)
        if can is None:
            stats["rejected_standardise"] += 1
            continue
        s = to_selfies_checked(can)
        if s is None:
            stats["rejected_roundtrip"] += 1
            continue
        kept_smi.append(can)
        kept_sf.append(s)
        if report_every and (i + 1) % report_every == 0:
            print(f"  processed {i+1:,} kept {len(kept_smi):,}")

    # deduplicate, preserving order
    seen, u_smi, u_sf = set(), [], []
    for smi, s in zip(kept_smi, kept_sf):
        if smi in seen:
            stats["duplicates"] += 1
            continue
        seen.add(smi)
        u_smi.append(smi)
        u_sf.append(s)

    stats["kept"] = len(u_smi)
    stats["roundtrip_failure_rate"] = (
        stats["rejected_roundtrip"] / max(1, stats["seen"] - stats["rejected_standardise"])
    )
    return u_smi, u_sf, dict(stats)


# ---------------------------------------------------------------- scaffold split
def murcko_scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(smiles: List[str],
                   val_frac: float = 0.05,
                   test_frac: float = 0.05,
                   seed: int = 0) -> Dict[str, np.ndarray]:
    """Group by Bemis-Murcko scaffold and assign whole groups to splits, so no
    scaffold appears in more than one split. This matters more than usual here:
    Phase 2 contrastive extraction sets are drawn from these splits, and a
    shared scaffold would let a steering direction encode a memorised motif."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(idx)

    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    # largest groups first keeps split sizes close to target
    keys.sort(key=lambda k: -len(groups[k]))

    n = len(smiles)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    test, val, train = [], [], []
    for k in keys:
        if len(test) < n_test:
            test.extend(groups[k])
        elif len(val) < n_val:
            val.extend(groups[k])
        else:
            train.extend(groups[k])

    return {"train": np.array(sorted(train)),
            "val": np.array(sorted(val)),
            "test": np.array(sorted(test))}

# ---------------------------------------------------------------- driver
def prepare(smiles_list: Iterable[str],
            out_dir: str,
            n_prefix: int = 8,
            max_len: int = 128,
            val_frac: float = 0.05,
            test_frac: float = 0.05,
            seed: int = 0):
    

    os.makedirs(out_dir, exist_ok=True)
    print("[1/5] standardising and round-trip checking ...")
    smi, selfies, stats = build_corpus(smiles_list)
    print(f"      kept {stats['kept']:,} / {stats['seen']:,}  "
          f"round-trip failure rate {stats['roundtrip_failure_rate']:.4%}")
    if stats["roundtrip_failure_rate"] > 0.01:
        print("      WARNING: round-trip failure >1%. Inspect standardisation "
              "before proceeding — do not attribute this to the model later.")

    print("[2/5] scaffold splitting ...")
    splits = scaffold_split(smi, val_frac, test_frac, seed)
    for k, v in splits.items():
        print(f"      {k}: {len(v):,}")

    print("[3/5] building alphabet from TRAIN split only ...")
    train_sf = [selfies[i] for i in splits["train"]]
    tok = SelfiesTokenizer.build(train_sf, n_prefix=n_prefix, max_len=max_len)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"      vocab size {tok.vocab_size} ({len(tok.symbols)} SELFIES symbols)")

    print("[4/5] tokenising ...")
    encoded: Dict[str, np.ndarray] = {}
    kept_smiles: Dict[str, List[str]] = {}
    for name, idx in splits.items():
        rows, keep = [], []
        for i in idx:
            ids = tok.encode(selfies[i])
            if ids is None:
                continue                   # too long, or OOV in val/test
            rows.append(ids)
            keep.append(smi[i])
        encoded[name] = np.asarray(rows, dtype=np.int16)
        kept_smiles[name] = keep
        print(f"      {name}: {len(rows):,} sequences")

    print("[5/5] caching ...")
    for name in encoded:
        np.save(os.path.join(out_dir, f"{name}.npy"), encoded[name])
        with open(os.path.join(out_dir, f"{name}_smiles.txt"), "w") as f:
            f.write("\n".join(kept_smiles[name]))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return tok, encoded, stats


# ---------------------------------------------------------------- gate 1
def gate1_verify(tokenizer, encoded: np.ndarray, smiles: List[str],
                 n: int = 1000, seed: int = 0) -> float:
    """GATE 1: decode n random tokenised rows and require exact SMILES match.
    Anything below 100% means the pipeline is lossy. Stop and fix it."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(encoded), size=min(n, len(encoded)), replace=False)
    ok = 0
    for i in idx:
        rec = tokenizer.decode_smiles(encoded[i])
        if rec is None:
            continue
        m = Chem.MolFromSmiles(rec)
        if m and Chem.MolToSmiles(m, canonical=True) == smiles[i]:
            ok += 1
    rate = ok / len(idx)
    print(f"GATE 1 round-trip through tokenizer: {rate:.4%} ({ok}/{len(idx)})")
    if rate < 1.0:
        print("  FAIL — do not proceed to training.")
    return rate