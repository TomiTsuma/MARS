# MARS — Masked Diffusion Language Model for Molecules


ARIES can function as an acronym by mapping each letter to core technical components of the method: Attention-based Representation Intervention and Elicitation of Structure for conditional molecule generation.Acronym BreakdownA – Attention-based (referring to the cross-attention or self-attention mechanisms in the diffusion backbone)R – Representation (targeting the latent feature spaces or hidden states)I – Intervention (describing the active steering or modification of those internal vectors)E – Elicitation (drawing out or inducing desired chemical properties)S – Structure (referring to the 3D molecular geometry or graph topology being generated)


Unconditional molecule generation with an absorbing-state (masked) discrete
diffusion language model, trained on ZINC250k in SELFIES representation.
SELFIES guarantees every decoded sequence is a chemically valid molecule, so
the model never has to learn syntactic validity — only chemistry.

Architecture: a pre-LN Transformer (`model/model.py`) with adaLN-zero time
conditioning, RoPE, and RMSNorm, trained against the continuous-time NELBO
from Austin et al.'s D3PM / Sahoo et al.'s MDLM formulation
(`model/objective.py`, `model/schedule.py`). Every block also exposes two
inert hook sites (`model/hooks.py`) for later representation-steering work —
irrelevant to training and sampling, safe to ignore for now.

## Setup

```
pip install -r requirements.txt
```

Requires a CUDA GPU for training at any real scale (`--device cpu` works for
smoke tests). Verify your install with:

```
python -c "import torch; print(torch.cuda.is_available())"
```

## Pipeline

Three stages, each a script under `scripts/`, each consuming the previous
stage's output:

```
data/250k_rndm_zinc_drugs_clean_3.csv
        |
        v
scripts/prepare.py  --> artifacts/processed/{tokenizer.json, train.npy, val.npy, test.npy, ...}
        |
        v
scripts/train.py    --> runs/<out_dir>/{config.json, ckpt_*.pt, latest.pt, train_log.jsonl}
        |
        v
scripts/sample.py   --> runs/<out_dir>/samples.smi  (+ optional metrics report)
```

### 1. Prepare the data

```
python scripts/prepare.py
```

Reads `data/250k_rndm_zinc_drugs_clean_3.csv`, and per molecule:

1. standardises the SMILES (canonical, desalted, neutral, restricted to a
   fixed element set, ≤60 heavy atoms),
2. converts to SELFIES and verifies the round trip decodes back to the
   *same* canonical SMILES (`GATE 1` — anything below 100% means the
   pipeline is lossy and training should not proceed),
3. splits by Bemis-Murcko scaffold (`train`/`val`/`test`, 90/5/5 by default)
   so no scaffold leaks across splits,
4. builds the SELFIES token vocabulary from the **train split only**, and
5. tokenises and caches everything under `artifacts/processed/`.

Useful flags: `--csv`, `--out-dir`, `--val-frac`/`--test-frac`, `--seed`.
Run `python scripts/prepare.py --help` for the full list.

Output you should see land in `artifacts/processed/`:

| file | contents |
|---|---|
| `tokenizer.json` | vocabulary + special tokens ([PAD]/[MASK]/[BOS]/prefix) |
| `train.npy`, `val.npy`, `test.npy` | tokenised, padded sequences (int16) |
| `train_smiles.txt`, `val_smiles.txt`, `test_smiles.txt` | canonical SMILES, same row order as the `.npy` files |
| `stats.json` | how many molecules were seen/kept/rejected, round-trip failure rate |

### 2. Train

```
python scripts/train.py
```

Config comes from `config/config.py` (`Config()` = Config-S, ~108M params,
d_model=640/12 layers; `--config-b` swaps in Config-B, d_model=768/14
layers). **`ModelConfig` fields are IRREVERSIBLE** — they're baked into the
trained weights, so pick the architecture before a long run.

Common flags (see `python scripts/train.py --help`):

| flag | purpose |
|---|---|
| `--config-b` | train Config-B instead of Config-S |
| `--out-dir` | where checkpoints/logs go (default `runs/phase1`) |
| `--batch-size`, `--lr`, `--max-steps`, `--warmup-steps` | override `TrainConfig` |
| `--precision {bf16,fp16,fp32}` | default `bf16` |
| `--resume PATH` | resume from a checkpoint (see below) |
| `--device`, `--num-workers`, `--seed` | environment knobs |

**The model is saved automatically.** Every `--ckpt-every` steps (and at the
final step), the script writes:

- `runs/<out_dir>/ckpt_<step>.pt` — a numbered snapshot
- `runs/<out_dir>/latest.pt` — always the most recent checkpoint (this is
  what `--resume` and `scripts/sample.py` expect by default)

Each checkpoint contains the raw model weights **and** an EMA (exponential
moving average) shadow copy. Sampling and evaluation always default to the
EMA weights — noisier raw weights are kept in the checkpoint for
resuming/debugging only.

`runs/<out_dir>/config.json` is written once at the start of the run (the
fully-resolved config, for exact reproduction) and `train_log.jsonl` gets one
JSON record per logged/evaluated step (train loss, val loss on both raw and
EMA weights, accuracy, grad norm, learning rate, throughput) — tail it or
load it with `pandas.read_json(..., lines=True)` to plot curves.

To resume:

```
python scripts/train.py --resume runs/phase1/latest.pt
```

`--resume` without `--config` automatically loads the `config.json` sitting
next to the checkpoint, so the architecture can't silently drift between
runs. Pass `--config path/to/config.json` explicitly if you want to resume
into a different directory than the one the checkpoint lives in.

**Smoke test before committing to a long run:**

```
python scripts/train.py --out-dir runs/smoke --batch-size 8 --max-steps 20 \
    --log-every 5 --eval-every 10 --ckpt-every 20 --num-workers 0
```

### 3. Sample

```
python scripts/sample.py --checkpoint runs/phase1/latest.pt --n-samples 10000
```

Loads the config from `config.json` next to the checkpoint (same convention
as `--resume`), samples via ancestral reverse diffusion
(`model/sampler.py`), decodes SELFIES back to SMILES, and writes one SMILES
per line to `runs/phase1/samples.smi` (override with `--out`).

Add `--eval` to also score the generated set against `train`/`test_smiles.txt`
from `artifacts/processed/` — validity, uniqueness, novelty, internal
diversity, nearest-neighbour similarity, scaffold similarity, Fréchet
ChemNet Distance, and QED/logP/MW summary stats (`model/metrics.py`), plus
the **Tier 0 gate**: validity ≥ 0.99, uniqueness ≥ 0.95, FCD ≤ 1.5. This gate
is the signal that the model is trained well enough to trust for any
downstream (e.g. representation-steering) work.

Other flags: `--n-steps` (reverse-diffusion steps, default from config),
`--temperature`, `--mode {ancestral,confidence}`, `--use-raw` (sample from
raw weights instead of EMA), `--seed`.

## Project structure

```
config/config.py       Config-S / Config-B dataclasses (data, model, diffusion, train)
datasets/tokenizer.py  SELFIES <-> token ids, vocabulary layout
datasets/dataset.py    Dataset / DataLoader, maskable/valid position masks
model/model.py         MDLM: DiT-style blocks, adaLN-zero, RoPE, weight-tied head
model/schedule.py      Noise schedules (log-linear default, cosine control arm)
model/objective.py     Forward masking process + NELBO loss (SUBS parameterisation)
model/sampler.py       Ancestral / confidence reverse sampling
model/metrics.py       Generative evaluation + the Tier 0 gate
model/hooks.py         Inert extraction/steering hook sites (future work)
scripts/prepare.py     CSV -> standardised, tokenised, split, cached data
scripts/train.py       Training loop, checkpointing, EMA, resume
scripts/sample.py      Load a checkpoint, generate, optionally evaluate
```
