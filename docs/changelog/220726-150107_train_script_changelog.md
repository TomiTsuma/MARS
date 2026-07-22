# train.py — unconditional MDLM training on ZINC250k (SELFIES)

**Date:** 2026-07-22
**File added:** `scripts/train.py`

## What this does

Training loop for the `MDLM` model (`model/model.py`) against the
continuous-time NELBO objective in `model/objective.py`, on the ZINC250k
SELFIES corpus produced by `scripts/prepare.py`. Unconditional generation
only — no property conditioning.

Reads `Config` from `config/config.py` (Config-S by default, `--config-b`
for the ~90M variant), loads `{tokenizer.json, train.npy, val.npy}` from
`--processed-dir`, and runs to `train.max_steps`.

## Design decisions worth flagging

- **EMA is not defined anywhere else in the repo**, despite
  `TrainConfig.ema_decay` and the explicit comment in `model/metrics.py`
  ("evaluate the EMA copy, always"). Added a minimal `EMA` class local to
  `train.py`. Validation is run against both raw and EMA weights each
  `eval_every` (by swapping the EMA shadow into the live model and back —
  avoids holding a second full model in memory).
- **AdamW parameter groups**: weight decay applies only to ≥2D tensors
  (linear/embedding weights), not to RMSNorm gains, biases, or the
  adaLN-zero projection biases. Standard practice; the objective docstring's
  emphasis on training stability (t_eps, low-discrepancy sampling) made this
  seem worth doing correctly rather than decaying everything uniformly.
- **LR schedule**: linear warmup (`train.warmup_steps`) → cosine decay to 0
  at `train.max_steps`, via `LambdaLR`. Not configurable beyond that; adding
  a schedule-type knob felt premature given `DiffusionConfig` already has an
  explicit `schedule` field for the noise process and no equivalent exists
  yet for the LR schedule.
- **Resume safety**: `ModelConfig` fields are marked IRREVERSIBLE in
  `config/config.py` (baked into checkpoint weights). If `--resume` is
  passed without `--config`, the script refuses to silently fall back to
  `Config()` defaults — it loads `config.json` saved next to the checkpoint,
  or raises if that file is missing. A silent mismatch here (e.g. resuming
  a Config-B checkpoint under Config-S dims) would corrupt the run without
  any visible error at load time.
- **Precision**: supports `bf16` (default, no grad scaler needed), `fp16`
  (with `GradScaler`), and `fp32`. `torch.compile` is available via
  `--compile` but defaults off — flagged in the script as unreliable on
  Windows (Triton backend).
- Every run writes `config.json` (full resolved config, for exact
  reproduction) and `train_log.jsonl` (one record per logged/evaluated
  step) to `--out-dir`, plus periodic `ckpt_{step}.pt` and a
  continuously-overwritten `latest.pt` for resume.

## Not tested end-to-end

This environment has no `torch` installed and no `artifacts/processed/`
data (i.e. `scripts/prepare.py` hasn't been run here), so I could only
verify the script parses (`python -m py_compile`) and cross-checked every
function signature it calls (`MDLM.__init__`, `training_step`,
`make_loader`, `SelfiesTokenizer.load`, `get_schedule`) against their
current definitions. Recommend a short smoke test — a few hundred steps
with a small `max_steps` override — before trusting a long run:

```
python scripts/prepare.py ...   # if artifacts/processed/ doesn't exist yet
python scripts/train.py --max-steps 200 --eval-every 100 --ckpt-every 100 --log-every 20
```
