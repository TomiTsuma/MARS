# prepare.py fix, sample.py, README.md, requirements.txt

**Date:** 2026-07-22

## Context

Follow-up to the earlier `train.py` addition. The user reported
`artifacts/processed/` was never being created after running
`scripts/prepare.py`. Root cause: `prepare.py` had `from __future__ import
annotations` placed after other imports (a hard `SyntaxError` in Python —
future imports must be the first statement in the file) and no CLI entry
point at all, so the script could never run. This had been flagged as a
known-but-out-of-scope issue in the previous session; it became blocking
once the user actually tried to run the pipeline.

## Changes

- **`scripts/prepare.py`** — moved the `__future__` import to line 1,
  deduplicated the import block, and added a `main()` / `if __name__ ==
  "__main__"` CLI driver: reads the CSV (`--csv`, `--smiles-col`, both
  defaulted to match `data/250k_rndm_zinc_drugs_clean_3.csv`), calls the
  existing `prepare()`/`gate1_verify()` functions, writes to `--out-dir`
  (default `artifacts/processed`, matching `DataConfig.processed_dir`).
- **`scripts/sample.py`** (new) — CLI wrapper around `model/sampler.py`'s
  `sample_to_smiles`, which previously had no entry point. Loads
  `config.json` next to the checkpoint (same convention `train.py --resume`
  uses), defaults to the **EMA** weights in the checkpoint (per the
  "evaluate the EMA copy, always" convention already documented in
  `config/config.py` and `model/metrics.py`), writes generated SMILES to
  `samples.smi`, and has an `--eval` flag that runs `model/metrics.evaluate`
  + `tier0_gate` against the train/test splits.
- **`requirements.txt`** — was empty despite the codebase depending on
  `torch`, `numpy`, `pandas`, `rdkit`, `selfies`, `fcd_torch` throughout.
  Populated with minimum versions matching what's installed and verified
  working in this environment (`torch==2.6.0+cu118`, `rdkit==2026.03.3`,
  etc.). This was in scope this time because the new README instructs `pip
  install -r requirements.txt`, and shipping that instruction against an
  empty file would be actively wrong documentation.
- **`README.md`** (new, repo root) — end-to-end pipeline docs:
  prepare → train → sample, with the exact commands, what each stage reads
  and writes, the checkpoint/EMA/resume contract, and what the `--eval`
  Tier 0 gate means.
- **`scripts/train.py`** — fixed a `FutureWarning` surfaced during
  verification: `torch.cuda.amp.GradScaler(...)` → `torch.amp.GradScaler(device.type,
  ...)` (deprecated call signature as of torch 2.6).

## Verification

This environment's default `python` lacked every dependency; `python3.11`
has the full stack installed, including CUDA-enabled torch. Used it to
actually run the pipeline rather than just read the code:

1. `python3.11 scripts/prepare.py` — already confirmed working from a prior
   run (168,130/249,455 molecules kept, 0% round-trip failure — see
   `artifacts/processed/stats.json`).
2. `python3.11 scripts/train.py --out-dir runs/smoke_test --batch-size 8
   --max-steps 20 ...` — ran to completion, loss decreased, checkpoint
   written.
3. `python3.11 scripts/sample.py --checkpoint runs/smoke_test/latest.pt
   --n-samples 20 --n-steps 16` — loaded the checkpoint, generated 20/20
   decodable SMILES (garbage chemistry, as expected from a 20-step smoke
   run — the point was exercising the checkpoint → sample path, not sample
   quality).

`runs/smoke_test/` was the throwaway output of step 2–3 above; left in
place (gitignored via `.gitignore` from the previous session) since cleanup
was denied.
