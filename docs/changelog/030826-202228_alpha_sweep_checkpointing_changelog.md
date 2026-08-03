# Alpha sweep checkpointing (`scripts/p2_alpha_sweep.py`)

## Why

`p2_alpha_sweep.py` runs for days: `len(alphas) x seeds x n` full diffusion
samples (default 17 x 3 x 10,000). Previously, nothing was persisted until the
very end of the script — a crash, OOM, preemption, or Ctrl-C at any point
before completion voided every molecule sampled so far.

## What changed

- **Checkpoint granularity**: one `(alpha, seed)` condition — the smallest
  unit of work that can be checkpointed without instrumenting
  `sample_to_smiles` itself.
- **Checkpoint location**: `artifacts/tmp/{property}/{seeds}/`
  - `state.json` — progress ledger: completed conditions, cached per-alpha
    sanity reports, and the args the run was started with. Written atomically
    (temp file + `os.replace`) so a crash mid-write can't corrupt the ledger
    the resume decision is based on (same pattern used to fix the `train.py`
    checkpoint corruption issue from 2026-07-25).
  - `rows.jsonl` — one JSON record per generated molecule, append-only. A
    truncated trailing line from a mid-write crash is dropped on load instead
    of corrupting the whole file.
  - `smiles_NNN.txt` — raw (pre-RDKit-filter) generated SMILES per alpha
    index, append-only. Needed because `sanity_report()` requires the
    unfiltered list to compute `selfies_validity`; `rows.jsonl` alone only has
    the RDKit-parseable subset and can't reconstruct it.
  - `RESUME_INSTRUCTIONS.txt` — always present once the first condition
    finishes, rewritten after every condition. Plain-English status (X/Y
    conditions done, which alphas are complete/in-progress) plus the exact
    command to re-run to resume. Says `SWEEP COMPLETE` once finished.
- **Auto-resume**: re-running the same command detects the checkpoint,
  validates that the arguments that affect reproducibility (`--data`,
  `--ckpt`, `--property`, `--alpha`, `--n`, `--seeds`, `--n-steps`,
  `--position-spec`, `--schedule-spec`, `--artifacts`, `--config`, and the
  resolved steering `direction_id`) are unchanged, and skips every condition
  already completed. A mismatch on any of those raises a clear error rather
  than silently mixing conditions sampled under different settings.
- **`--restart`**: archives (renames, does not delete) any existing
  checkpoint for the same `{property}/{seeds}` and starts over — an explicit,
  non-destructive way to discard progress instead of deleting files by hand.
- The sweep loop is wrapped so any exception (including `KeyboardInterrupt`)
  prints the path to `RESUME_INSTRUCTIONS.txt` before re-raising.

## Not changed

- Final artifacts (`generations_{property}.parquet`, `pareto_{property}.csv`,
  `e2_{property}.json`) still write to `--artifacts` (default `artifacts/p2`)
  as before — the `artifacts/tmp/...` checkpoint is scratch/resume state, not
  a new output location.
- Ring-system reference building (`build_training_ring_systems`, the
  `train_rings.json` cache step) was left as-is — it already caches to disk
  once, and checkpointing it mid-build would require changes inside
  `eval/chemical_sanity.py` that are out of scope here. The days-long cost is
  the sampling loop, not ring-system construction.

## Testing

Verified with `python -m py_compile scripts/p2_alpha_sweep.py`. Not yet run
end-to-end against a real checkpoint/dataset — recommend a smoke test with
small `--n` and `--seeds` (and deliberately killing the process mid-run) to
confirm the resume path before trusting it for a multi-day run.
