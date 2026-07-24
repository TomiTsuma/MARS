"""
Direction artefacts and their store.

Every steering direction is stored with enough provenance to reconstruct
exactly how it was produced. This is not bookkeeping: a direction is a claim
about the model's internal representation, and a claim that cannot be traced
to a procedure is not reportable.

`projection_spearman` is a required field. It is the cheapest validity check
available (does projecting held-out molecules onto this direction correlate
with the property?) and storing it on the artefact means no direction can be
used downstream without its own evidence attached.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DirectionArtifact:
    vector: np.ndarray                # (d_model,) L2-normalised

    # what it steers
    property: str
    # where it came from
    layer: int
    site: str                         # "prefix" | "pooled" | "masked"
    position: int = 0                 # index within the site
    estimator: str = "diffmeans"      # "diffmeans" | "ridge"
    noise_level: float = 0.0          # extraction diffusion time t

    # data provenance
    corpus: str = ""
    split: str = ""
    n_samples: int = 0
    n_strata: int = 0
    seed: int = 0

    # evidence
    projection_spearman: float = float("nan")
    bootstrap_cos_std: float = float("nan")
    heavy_mean_abs_diff: float = float("nan")   # Gate A residual size leak

    # model provenance
    model_ckpt: str = ""
    model_step: int = -1
    created_at: str = ""

    extra: Dict = field(default_factory=dict)

    # ---------------------------------------------------------------- id
    @property
    def id(self) -> str:
        key = "|".join(str(x) for x in (
            self.property, self.layer, self.site, self.position,
            self.estimator, self.noise_level, self.corpus, self.split,
            self.n_samples, self.seed, self.model_ckpt))
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def is_usable(self, min_rho: float = 0.3) -> bool:
        """A direction whose held-out projection does not track the property is
        noise. No amount of coefficient tuning rescues it — return to
        extraction rather than to alpha."""
        r = self.projection_spearman
        return np.isfinite(r) and abs(r) >= min_rho

    def meta(self) -> Dict:
        d = asdict(self)
        d.pop("vector")
        d["id"] = self.id
        d["dim"] = int(self.vector.shape[-1])
        d["norm"] = float(np.linalg.norm(self.vector))
        return d


class DirectionStore:
    """Filesystem-backed store: one .npy per direction, one manifest.json."""

    def __init__(self, root: str = "artifacts/p2/directions"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.manifest_path = os.path.join(root, "manifest.json")
        self.manifest: Dict[str, Dict] = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)

    def save(self, art: DirectionArtifact) -> str:
        import datetime
        if not art.created_at:
            art.created_at = datetime.datetime.now().isoformat(timespec="seconds")
        aid = art.id
        np.save(os.path.join(self.root, f"{aid}.npy"),
                np.asarray(art.vector, dtype=np.float32))
        self.manifest[aid] = art.meta()
        self._flush()
        return aid

    def load(self, aid: str) -> DirectionArtifact:
        meta = dict(self.manifest[aid])
        meta.pop("id", None); meta.pop("dim", None); meta.pop("norm", None)
        vec = np.load(os.path.join(self.root, f"{aid}.npy"))
        return DirectionArtifact(vector=vec, **meta)

    def query(self, **filters) -> List[Dict]:
        out = []
        for aid, m in self.manifest.items():
            if all(m.get(k) == v for k, v in filters.items()):
                out.append({**m, "id": aid})
        return out

    def best(self, property: str, min_rho: float = 0.3) -> Optional[DirectionArtifact]:
        cand = [m for m in self.query(property=property)
                if abs(m.get("projection_spearman", 0) or 0) >= min_rho]
        if not cand:
            return None
        best = max(cand, key=lambda m: abs(m["projection_spearman"]))
        return self.load(best["id"])

    def _flush(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def __len__(self):
        return len(self.manifest)