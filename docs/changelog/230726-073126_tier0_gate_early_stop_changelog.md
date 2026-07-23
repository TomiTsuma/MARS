# Tier 0 gate early-stopping in train.py

## What changed

`scripts/train.py` now checks the Tier 0 gate periodically during training
and stops early (saving a distinct checkpoint) the first time it passes,
instead of always training to `max_steps`.

- **`config/config.py`**: added `TrainConfig.gate_every` (default `10_000`)
  and `TrainConfig.gate_n_samples` (default `1_000`).
- **`scripts/train.py`**:
  - `load_gate_smiles(processed_dir)` — loads `train_smiles.txt` /
    `test_smiles.txt` (already written by `prepare.py`), used as the
    reference sets for gate scoring.
  - `sample_gate_smiles(...)` — mirrors `evaluate_ema`: swaps the EMA shadow
    weights into the live model, calls `model.sampler.sample_to_smiles`,
    swaps the raw weights back. The EMA copy is judged, consistent with
    every other eval path in this file and with `scripts/sample.py`.
  - In the training loop, every `gate_every` steps: sample
    `gate_n_samples` molecules from the EMA weights, score them with
    `model.metrics.evaluate`, log the report as a `"gate"` split record in
    `train_log.jsonl`, then run `model.metrics.tier0_gate`. On the first
    pass, save the checkpoint to `<out_dir>/gate_passed.pt` and `break` out
    of the loop.
  - Also fixed a pre-existing `SyntaxError` in `config/config.py`
    (`TrainConfig.out_dir`'s default used nested double quotes inside an
    f-string, which only Python 3.12+ accepts) — this blocked `train.py`
    from importing at all under the project's Python 3.11 environment used
    for testing.
  - Final "done" message now reports the actual step count instead of
    always printing `max_steps`, since training may now stop early.
  - New CLI flags: `--gate-every`, `--gate-n-samples`.

## Why

Phase 1 pretraining doesn't need to run to a fixed step budget — the
purpose is to reach Tier 0 (validity/uniqueness/FCD) and stop, since
`model/metrics.py` already documents the gate as the thing that must pass
before Phase 2 steering work begins. Baking the check into the training
loop means a run self-terminates as soon as its EMA weights clear the bar,
rather than requiring someone to babysit `train_log.jsonl` or run
`sample.py --eval` manually after the fact.

## Verification

Ran two smoke tests with `py -3.11 scripts/train.py` against the existing
`artifacts/processed` data (small `--max-steps`, `--gate-every 2`,
`--gate-n-samples 8`, `--batch-size 8`):

1. Normal run — the gate check fired at steps 2 and 4, printed the
   `TIER 0 GATE` report (FCD correctly failing on a near-random model),
   logged a `"gate"` split record with the full metrics report each time,
   and training continued to `max_steps` as normal.
2. Forced-pass run — monkeypatched `metrics.tier0_gate` to return `True` to
   verify the stop path without needing a fully trained model. Training
   stopped at step 2 (the first `gate_every` boundary), `gate_passed.pt`
   was written to the run's `out_dir`, and the final log line reported
   `done: 2 steps` instead of the configured `max_steps`.

Both scratch run directories (`runs/gate_smoke_test`, `runs/gate_smoke_pass`)
are gitignored; not committed.
