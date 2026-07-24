"""
Content-addressed property cache.

Without this, property computation dominates evaluation wall-clock. Phase 2
generates on the order of 1.8M molecules across all sweeps, and many recur
between conditions (the unconditional baseline especially). Keying on
canonical SMILES means a molecule is scored once no matter how often it is
regenerated.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence

from properties.oracles.rdkit_props import DEFAULT_PROPS, canonical, compute


class PropertyCache:
    """SQLite-backed cache. Chosen over an in-memory dict because sweeps run
    as separate processes and must share results; and over parquet because
    lookups are random-access rather than scans."""

    def __init__(self, path: str = "artifacts/prop_cache.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS props ("
            "  smiles TEXT PRIMARY KEY,"
            "  payload TEXT NOT NULL)"
        )
        self.conn.commit()
        self._hits = 0
        self._misses = 0

    # ---- single ---------------------------------------------------------
    def get(self, smiles: str) -> Optional[Dict]:
        can = canonical(smiles)
        if can is None:
            return None
        row = self.conn.execute(
            "SELECT payload FROM props WHERE smiles = ?", (can,)).fetchone()
        if row is None:
            self._misses += 1
            return None
        self._hits += 1
        return json.loads(row[0])

    def put(self, rec: Dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO props (smiles, payload) VALUES (?, ?)",
            (rec["smiles"], json.dumps(rec)))

    # ---- bulk -----------------------------------------------------------
    def bulk_annotate(self, smiles_list: Iterable[str],
                      props: Sequence[str] = DEFAULT_PROPS,
                      commit_every: int = 5000):
        """Return a DataFrame for `smiles_list`, computing only cache misses."""
        import pandas as pd

        smiles_list = list(smiles_list)
        out: List[Optional[Dict]] = []
        n_new = 0
        for i, smi in enumerate(smiles_list):
            rec = self.get(smi)
            if rec is None:
                rec = compute(smi, props)
                if rec is not None:
                    self.put(rec)
                    n_new += 1
            out.append(rec)
            if n_new and n_new % commit_every == 0:
                self.conn.commit()
        self.conn.commit()
        return pd.DataFrame([r for r in out if r is not None])

    def warm_from_frame(self, df) -> int:
        """Seed the cache from an already-annotated corpus parquet."""
        n = 0
        for rec in df.to_dict("records"):
            self.put(rec)
            n += 1
        self.conn.commit()
        return n

    @property
    def stats(self) -> Dict[str, int]:
        n = self.conn.execute("SELECT COUNT(*) FROM props").fetchone()[0]
        return {"rows": n, "hits": self._hits, "misses": self._misses}

    def close(self):
        self.conn.commit()
        self.conn.close()