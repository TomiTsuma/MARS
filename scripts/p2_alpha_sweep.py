"""
Phase 2, Stage E — the alpha sweep (E2). THE HEADLINE EXPERIMENT.

    python3.11 scripts/p2_alpha_sweep.py  --data artifacts/processed --ckpt runs/phase1-configb/latest.pt  --property logp --n 10000 --seeds 3


Produces:
    Figure B1  dose-response with Spearman rho   (Tier 1 criterion: rho > 0.7)
    Figure B2  Pareto frontier                    (THE usefulness verdict)
    Figure E   validity vs chemical sanity        (the SELFIES trap check)

The sweep includes NEGATIVE coefficients deliberately. Bidirectional control is
part of the claim: if negative alpha does not reduce the property, that is
evidence about linearity, not a bug to tune away.

Reporting protocol here is the full one — 10,000 molecules, N=128, three seeds.
Do not use the reduced sweep budget for these numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, ".")
from config.config import Config
from datasets.tokenizer import SelfiesTokenizer
from model.schedule import get_schedule
from model.sampler import sample_to_smiles
from model.model import MDLM
from properties.oracles.rdkit_props import compute, EXACT_TARGETS, DEFAULT_PROPS
from properties.steering import (load_residual_stats, DirectionStore,
                                  AdditiveSteer, schedule_from_spec,
                                  position_from_spec, monotonicity)
from eval.chemical_sanity import (build_training_ring_systems,
                                              sanity_report, trap_check)
from eval.steering_metrics import (spearman_dose, property_shift,
                                               pareto_frontier)


def parse_alpha(spec: str):
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


# ------------------------------------------------------------------ checkpointing
#
# The sweep is O(days): len(alphas) x seeds x n molecules, each condition costing
# a full diffusion sample. Nothing below is optional polish -- without it, any
# crash, OOM, preemption, or Ctrl-C after hour 40 loses every molecule sampled so
# far. Checkpoint granularity is one (alpha, seed) condition: the smallest unit
# of work whose loss is tolerable, since going finer would mean instrumenting
# sample_to_smiles itself.
#
# Layout inside artifacts/tmp/{property}/{seeds}/:
#   state.json       - progress ledger (completed conditions, cached per-alpha
#                       sanity reports, the args this run was started with).
#                       Rewritten atomically (tmp file + os.replace) since it is
#                       read back to decide what work remains.
#   rows.jsonl        - one JSON object per generated molecule, append-only. A
#                       truncated last line (crash mid-write) is dropped on load
#                       instead of corrupting the whole file.
#   smiles_NNN.txt    - raw non-empty generated SMILES for alpha index NNN,
#                       append-only. This is the pre-RDKit-filtering list that
#                       sanity_report() needs to compute selfies_validity
#                       correctly; rows.jsonl alone only has the parseable
#                       subset, so it can't be reconstructed from rows.
#   RESUME_INSTRUCTIONS.txt - always present once the first condition finishes,
#                       rewritten after every condition. Human-readable: what's
#                       done, what's left, and the exact command to resume.

CRITICAL_ARGS = ["data", "ckpt", "config", "artifacts", "property", "alpha",
                  "n", "seeds", "n_steps", "position_spec", "schedule_spec"]


def checkpoint_dir_for(args) -> str:
    return os.path.join("artifacts", "tmp", args.property, str(args.seeds))


def _paths(checkpoint_dir):
    return {
        "state": os.path.join(checkpoint_dir, "state.json"),
        "rows": os.path.join(checkpoint_dir, "rows.jsonl"),
        "instructions": os.path.join(checkpoint_dir, "RESUME_INSTRUCTIONS.txt"),
    }


def _smiles_path(checkpoint_dir, alpha_idx):
    return os.path.join(checkpoint_dir, f"smiles_{alpha_idx:03d}.txt")


def atomic_write_json(path, obj):
    """Write-to-temp-then-replace so a crash mid-write can never corrupt the
    ledger the resume decision is based on."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, records):
    if not records:
        return
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_smiles(checkpoint_dir, alpha_idx, smis):
    if not smis:
        return
    with open(_smiles_path(checkpoint_dir, alpha_idx), "a") as f:
        for s in smis:
            f.write(s + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_smiles(checkpoint_dir, alpha_idx):
    path = _smiles_path(checkpoint_dir, alpha_idx)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def load_rows(checkpoint_dir):
    path = _paths(checkpoint_dir)["rows"]
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()
    rows = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print("[checkpoint] dropping truncated trailing row "
                      "(previous run crashed mid-write)")
                continue
            raise
    return rows


def load_state(checkpoint_dir, args, direction_id):
    """Returns None for a fresh run, or the validated prior state to resume
    from. Refuses to resume if the args that matter for reproducibility
    changed, since silently mixing conditions sampled under different
    settings would corrupt the sweep."""
    path = _paths(checkpoint_dir)["state"]
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    mismatches = []
    for key in CRITICAL_ARGS:
        old, new = state.get("args", {}).get(key), getattr(args, key)
        if old != new:
            mismatches.append(f"  --{key.replace('_', '-')}: "
                              f"checkpoint={old!r} vs this run={new!r}")
    if state.get("direction_id") not in (None, direction_id):
        mismatches.append(f"  direction_id: checkpoint={state['direction_id']!r} "
                          f"vs this run={direction_id!r}")
    if mismatches:
        raise SystemExit(
            f"[checkpoint] refusing to resume from {checkpoint_dir}: this run's "
            "settings differ from the checkpoint:\n" + "\n".join(mismatches) +
            "\n\nEither re-run with the original arguments, or pass --restart "
            "to archive the old checkpoint and start fresh."
        )
    return state


def save_state(checkpoint_dir, args, completed, per_alpha_sanity, direction_id,
               started_at):
    state = {
        "args": {k: getattr(args, k) for k in CRITICAL_ARGS},
        "direction_id": direction_id,
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed": sorted([a, s] for a, s in completed),
        "per_alpha_sanity": {f"{a:.4f}": rep for a, rep in per_alpha_sanity.items()},
    }
    atomic_write_json(_paths(checkpoint_dir)["state"], state)


def write_resume_instructions(checkpoint_dir, args, completed, alphas, started_at,
                              done_all=False):
    total = len(alphas) * args.seeds
    done = len(completed)
    per_alpha_done = {a: sum(1 for s in range(args.seeds) if (a, s) in completed)
                      for a in alphas}
    complete_alphas = [a for a, c in per_alpha_done.items() if c == args.seeds]
    partial = [(a, c) for a, c in per_alpha_done.items() if 0 < c < args.seeds]
    cmd = shlex.join([sys.executable] + sys.argv)

    lines = [
        "MARS Phase 2 Alpha Sweep -- checkpoint status",
        "=" * 60,
        f"Property     : {args.property}",
        f"Seeds        : {args.seeds}",
        f"Started      : {started_at}",
        f"Last update  : {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Progress: {done}/{total} (alpha, seed) conditions complete "
        f"({100 * done / max(1, total):.1f}%)",
        f"Fully completed alphas: {len(complete_alphas)}/{len(alphas)}",
    ]
    if partial:
        lines.append("In-progress alphas: " + ", ".join(
            f"{a:+.2f} ({c}/{args.seeds} seeds)" for a, c in partial))

    if done_all:
        lines += [
            "",
            "STATUS: SWEEP COMPLETE.",
            f"Final artifacts were written to: {args.artifacts}/",
            "  generations_<property>.parquet, pareto_<property>.csv, "
            "e2_<property>.json",
            "",
            "This checkpoint folder is no longer needed and can be deleted.",
        ]
    else:
        lines += [
            "",
            "HOW TO RESUME",
            "-------------",
            "If this run is interrupted (crash, OOM, preemption, Ctrl-C), just",
            "re-run the exact command below. The script detects this checkpoint",
            "automatically and skips every (alpha, seed) condition already done",
            "-- no sampling is repeated.",
            "",
            f"    {cmd}",
            "",
            f"Checkpoint folder: {checkpoint_dir}",
            "  state.json      - progress ledger + cached per-alpha sanity reports",
            "  rows.jsonl      - one JSON record per generated molecule",
            "  smiles_NNN.txt  - raw generated SMILES per alpha (sanity input)",
            "",
            "Do not hand-edit these files. Do not delete this folder until this",
            f"file says SWEEP COMPLETE and {args.artifacts}/ has the final outputs.",
            "To intentionally discard progress and start over, re-run with --restart",
            "instead of deleting things by hand.",
        ]
    with open(_paths(checkpoint_dir)["instructions"], "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--artifacts", default="artifacts/p2")
    ap.add_argument("--property", default="logp", choices=list(EXACT_TARGETS))
    ap.add_argument("--alpha", default="-4:4:0.5", help="lo:hi:step")
    ap.add_argument("--n", type=int, default=10000, help="molecules per condition")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--position-spec", default="all")
    ap.add_argument("--schedule-spec", default="all")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--restart", action="store_true",
                    help="Archive any existing checkpoint for this property/seeds "
                         "and start the sweep over from scratch.")
    args = ap.parse_args()

    checkpoint_dir = checkpoint_dir_for(args)
    if args.restart and os.path.isdir(checkpoint_dir):
        archived = f"{checkpoint_dir}.archived_{datetime.now():%Y%m%d-%H%M%S}"
        os.rename(checkpoint_dir, archived)
        print(f"[checkpoint] --restart: archived previous checkpoint to {archived}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg_path = args.config or os.path.join(os.path.dirname(args.ckpt), "config.json")
    cfg = Config.from_json(cfg_path)
    tok = SelfiesTokenizer.load(os.path.join(args.data, "tokenizer.json"))
    schedule = get_schedule(cfg.diffusion.schedule)

    model = MDLM(tok.vocab_size, cfg.model, pad_id=tok.pad_id).to(args.device)
    ck = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ck["model"])
    if "ema" in ck:
        sd = model.state_dict()
        for k, v in ck["ema"].items():
            sd[k].copy_(v.to(sd[k].dtype))
    model.eval()

    with open(os.path.join(args.artifacts, f"selected_{args.property}.json")) as f:
        sel = json.load(f)
    store = DirectionStore(os.path.join(args.artifacts, "directions"))
    art = store.load(sel["direction_id"])
    stats = load_residual_stats(os.path.join(args.artifacts, "residual_stats.pt"))
    print(f"[sweep] direction {art.id}  layer {art.layer} pos {art.position}  "
          f"rho={art.projection_spearman:+.3f}")
    if not art.is_usable():
        print("[sweep] WARNING: projection check below 0.3 — this direction may "
              "be noise. Consider returning to extraction rather than sweeping.")

    prior_state = load_state(checkpoint_dir, args, art.id)
    started_at = (prior_state["started_at"] if prior_state
                 else datetime.now().isoformat(timespec="seconds"))

    with open(os.path.join(args.data, "train_smiles.txt")) as f:
        train_smiles = [l.strip() for l in f if l.strip()]
    ring_cache = os.path.join(args.artifacts, "train_rings.json")
    if os.path.exists(ring_cache):
        train_rings = set(json.load(open(ring_cache)))
    else:
        print("[sweep] building training ring-system reference ...")
        train_rings = build_training_ring_systems(train_smiles)
        json.dump(sorted(train_rings), open(ring_cache, "w"))
    print(f"[sweep] {len(train_rings):,} reference ring systems")

    alphas = parse_alpha(args.alpha)
    print(f"[sweep] {len(alphas)} coefficients x {args.seeds} seeds "
          f"x {args.n} molecules @ N={args.n_steps}")

    rows = load_rows(checkpoint_dir)
    completed = {(a, s) for a, s in prior_state["completed"]} if prior_state else set()
    per_alpha_sanity = ({float(k): v for k, v in prior_state["per_alpha_sanity"].items()}
                        if prior_state else {})
    total_conditions = len(alphas) * args.seeds
    if completed:
        print(f"[checkpoint] resuming from {checkpoint_dir}: "
              f"{len(completed)}/{total_conditions} conditions already done")

    try:
        for a in alphas:
            alpha_idx = alphas.index(a)
            for seed in range(args.seeds):
                if (a, seed) in completed:
                    continue
                torch.manual_seed(seed)
                steer = (None if a == 0.0 else AdditiveSteer(
                    torch.from_numpy(art.vector), a, stats=stats,
                    layers={art.layer}, schedule=schedule_from_spec(args.schedule_spec),
                    positions=position_from_spec(args.position_spec),
                    n_struct=tok.n_struct, pad_id=tok.pad_id, mask_id=tok.mask_id))
                gen = sample_to_smiles(model, schedule, tok, args.n,
                                       n_steps=args.n_steps, device=args.device,
                                       batch_size=args.batch_size, steer=steer)
                valid = [s for s in gen if s]
                new_rows = []
                for s in valid:
                    r = compute(s)
                    if r:
                        new_rows.append({"alpha": a, "seed": seed, **r})
                rows.extend(new_rows)

                # persist this condition before moving on -- everything above this
                # line is lost on a crash, everything at or below it survives.
                append_jsonl(_paths(checkpoint_dir)["rows"], new_rows)
                append_smiles(checkpoint_dir, alpha_idx, valid)
                completed.add((a, seed))
                save_state(checkpoint_dir, args, completed, per_alpha_sanity,
                          art.id, started_at)
                write_resume_instructions(checkpoint_dir, args, completed, alphas,
                                          started_at)
                print(f"  [checkpoint] alpha={a:+.2f} seed={seed} done "
                      f"({len(completed)}/{total_conditions})")

            if a not in per_alpha_sanity:
                smis_all = read_smiles(checkpoint_dir, alpha_idx)
                per_alpha_sanity[a] = sanity_report(smis_all, train_rings)
                save_state(checkpoint_dir, args, completed, per_alpha_sanity,
                          art.id, started_at)
                write_resume_instructions(checkpoint_dir, args, completed, alphas,
                                          started_at)
            rep = per_alpha_sanity[a]
            vals = [r[args.property] for r in rows if r["alpha"] == a]
            print(f"  alpha={a:+.2f}  {args.property}={np.nanmean(vals):+.3f}  "
                  f"validity={rep['selfies_validity']:.3f}  "
                  f"sanity={rep['chemical_sanity']:.3f}")
    except BaseException:
        print(f"\n[sweep] interrupted -- progress through the last completed "
              f"condition is safely checkpointed. To resume, see:\n"
              f"    {_paths(checkpoint_dir)['instructions']}")
        raise

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(args.artifacts, f"generations_{args.property}.parquet"),
                  index=False)

    # ---------------------------------------------------------------- B1
    agg = df.groupby("alpha")[args.property].mean().reset_index()
    dose = spearman_dose(agg["alpha"], agg[args.property])
    base = df[df.alpha == 0.0][args.property]
    print(f"\n[B1] Spearman rho(alpha, {args.property}) = {dose['spearman']:+.3f}  "
          f"-> Tier 1 {'PASS' if dose['passes_tier1'] else 'FAIL'}")
    print(f"     linearity gap (|rho_s| - |rho_p|) = {dose['linearity_gap']:+.3f}  "
          f"(large positive = monotone but non-linear)")

    # ---------------------------------------------------------------- B2
    pareto_rows = []
    for a in alphas:
        sub = df[df.alpha == a]
        if sub.empty:
            continue
        ps = property_shift(sub[args.property], base, n_boot=200)
        pareto_rows.append({"alpha": a, "delta_property": ps["shift"],
                            "chemical_sanity": per_alpha_sanity[a]["chemical_sanity"],
                            "fidelity_cost": 1.0 - per_alpha_sanity[a]["chemical_sanity"],
                            "heavy_shift": float(sub.heavy.mean() - base.index.size * 0)
                            if "heavy" in sub else float("nan")})
    pdf = pd.DataFrame(pareto_rows)
    front = pareto_frontier(pdf, x="delta_property", y="fidelity_cost")
    pdf.to_csv(os.path.join(args.artifacts, f"pareto_{args.property}.csv"), index=False)
    print(f"[B2] Pareto frontier: {len(front)} non-dominated operating points")

    # ---------------------------------------------------------------- E
    tc = trap_check(per_alpha_sanity)
    print(f"\n[E]  validity range={tc['validity_range']:.3f}  "
          f"sanity range={tc['sanity_range']:.3f}")
    print(f"     {tc['message']}")

    with open(os.path.join(args.artifacts, f"e2_{args.property}.json"), "w") as f:
        json.dump({"dose_response": dose, "trap_check": tc,
                   "sanity_by_alpha": {str(k): v for k, v in per_alpha_sanity.items()},
                   "direction_id": art.id}, f, indent=2)
    print(f"\n[done] artefacts in {args.artifacts}/")

    write_resume_instructions(checkpoint_dir, args, completed, alphas, started_at,
                              done_all=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())