"""
Phase 2, Stage E — the alpha sweep (E2). THE HEADLINE EXPERIMENT.

    python scripts/p2_alpha_sweep.py \
        --data artifacts/zinc250k --ckpt runs/phase1/ckpt_final.pt \
        --property logp --alpha -4:4:0.5 --n 10000 --seeds 3

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
import sys

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
    args = ap.parse_args()

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

    rows, per_alpha_sanity = [], {}
    for a in alphas:
        smis_all = []
        for seed in range(args.seeds):
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
            smis_all.extend(valid)
            for s in valid:
                r = compute(s)
                if r:
                    rows.append({"alpha": a, "seed": seed, **r})
        rep = sanity_report(smis_all, train_rings)
        per_alpha_sanity[a] = rep
        vals = [r[args.property] for r in rows if r["alpha"] == a]
        print(f"  alpha={a:+.2f}  {args.property}={np.nanmean(vals):+.3f}  "
              f"validity={rep['selfies_validity']:.3f}  "
              f"sanity={rep['chemical_sanity']:.3f}")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())