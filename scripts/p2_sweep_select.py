"""
Phase 2, Stage C — candidate sweep and operating-point selection (E1 → Figure A).

    python scripts/p2_sweep_select.py \
        --data artifacts/zinc250k --ckpt runs/phase1/ckpt_final.pt \
        --property logp --artifacts artifacts/p2

Runs after p2_probe_sites.py, which supplies the directions, the residual
statistics and the chosen extraction site.

COST: uses a REDUCED generation budget (500 molecules, N=32, one seed) — about
15x cheaper per candidate than the reporting protocol. Sweeping at reporting
budget is the most common way this phase runs long.

Produces the layer-by-position sensitivity heatmap. Read the STRUCTURE, not
just the argmax: a tight hotspot means property information is localised, a
diffuse field means it is distributed. Both are findings.
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
from properties.oracles.rdkit_props import compute, EXACT_TARGETS
from properties.steering import (load_residual_stats, sweep_candidates,
                                  select_operating_point, heatmap_data,
                                  plot_heatmap, reference_fingerprints,
                                  DirectionArtifact, DirectionStore)


def load_model(ckpt, cfg, tok, device, use_ema=True):
    model = MDLM(tok.vocab_size, cfg.model, pad_id=tok.pad_id).to(device)
    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    if use_ema and "ema" in ck:
        sd = model.state_dict()
        for k, v in ck["ema"].items():
            sd[k].copy_(v.to(sd[k].dtype))
    model.eval()
    return model, ck.get("step", -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--artifacts", default="artifacts/p2")
    ap.add_argument("--property", default="logp", choices=list(EXACT_TARGETS))
    ap.add_argument("--estimator", default="diffmeans", choices=["diffmeans", "ridge"])
    ap.add_argument("--alpha-ref", type=float, default=2.0)
    ap.add_argument("--n-mols", type=int, default=500, help="REDUCED sweep budget")
    ap.add_argument("--n-steps", type=int, default=32, help="REDUCED sweep budget")
    ap.add_argument("--position-spec", default="all")
    ap.add_argument("--schedule-spec", default="all")
    ap.add_argument("--max-fcd-proxy", type=float, default=None)
    ap.add_argument("--max-heavy-shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(os.path.dirname(args.ckpt), "config.json")
    cfg = Config.from_json(cfg_path)
    tok = SelfiesTokenizer.load(os.path.join(args.data, "tokenizer.json"))
    schedule = get_schedule(cfg.diffusion.schedule)
    model, mstep = load_model(args.ckpt, cfg, tok, args.device)

    # ---- artefacts from Stage B ----------------------------------------
    with open(os.path.join(args.artifacts,
                           f"probe_report_{args.property}.json")) as f:
        report = json.load(f)
    site = report["decision"]["primary_site"]
    dpath = os.path.join(args.artifacts,
                         f"directions_{args.property}_{site}.npz")
    blob = np.load(dpath)
    directions = torch.from_numpy(blob[args.estimator]).float()
    stats = load_residual_stats(os.path.join(args.artifacts, "residual_stats.pt"))
    print(f"[sweep] site '{site}' | directions {tuple(directions.shape)} "
          f"| estimator {args.estimator}")

    # ---- reference distribution and unconditional baseline --------------
    with open(os.path.join(args.data, "train_smiles.txt")) as f:
        train_smiles = [l.strip() for l in f if l.strip()]
    ref_fps = reference_fingerprints(train_smiles, n=2000, seed=args.seed)

    print(f"[sweep] generating unconditional baseline ({args.n_mols} molecules)")
    torch.manual_seed(args.seed)
    base = sample_to_smiles(model, schedule, tok, args.n_mols,
                            n_steps=args.n_steps, device=args.device)
    base_valid = [s for s in base if s]
    base_props = [compute(s) for s in base_valid]
    base_props = [p for p in base_props if p]
    baseline_prop = float(np.nanmean([p[args.property] for p in base_props]))
    baseline_heavy = float(np.nanmean([p["heavy"] for p in base_props]))
    print(f"[sweep] baseline {args.property} = {baseline_prop:.3f} "
          f"| heavy = {baseline_heavy:.2f} | n_valid = {len(base_valid)}")

    def property_fn(smis):
        recs = [compute(s) for s in smis]
        recs = [r for r in recs if r]
        return {"target": np.array([r[args.property] for r in recs]),
                "heavy": np.array([r["heavy"] for r in recs]),
                "heavy_base": baseline_heavy}

    # ---- the sweep ------------------------------------------------------
    n_layers, n_sites, _ = directions.shape
    print(f"[sweep] evaluating {n_layers * n_sites} candidates "
          f"at alpha={args.alpha_ref} (reduced budget)")
    df = sweep_candidates(model, tok, schedule, directions, property_fn,
                          stats, ref_fps, baseline_prop,
                          alpha_ref=args.alpha_ref, n_mols=args.n_mols,
                          n_steps=args.n_steps,
                          position_spec=args.position_spec,
                          schedule_spec=args.schedule_spec,
                          device=args.device, seed=args.seed)
    out_csv = os.path.join(args.artifacts, f"sweep_{args.property}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[sweep] wrote {out_csv}")

    # ---- selection ------------------------------------------------------
    print("\n[select] applying constraints "
          f"(fcd_proxy<={args.max_fcd_proxy}, |heavy shift|<={args.max_heavy_shift})")
    best = select_operating_point(df, args.max_fcd_proxy,
                                  args.max_heavy_shift)
    if best is None:
        return 1

    # ---- Figure A -------------------------------------------------------
    Z = heatmap_data(df, n_layers, n_sites)
    labels = (["BOS"] + [f"P{i}" for i in range(n_sites - 1)]
              if site == "prefix" and n_sites > 1 else None)
    fig = plot_heatmap(Z, os.path.join(args.artifacts,
                                       f"figA_{args.property}.png"),
                       title=f"Layer x position sensitivity — {args.property} "
                             f"(site: {site}, alpha={args.alpha_ref})",
                       selected=best, position_labels=labels)
    print(f"[select] Figure A -> {fig}")

    finite = Z[np.isfinite(Z)]
    if finite.size:
        conc = float(np.abs(finite).max() / (np.abs(finite).mean() + 1e-8))
        print(f"""[select] concentration ratio (max/mean) = {conc:.2f}  
              ({'localised' if conc > 3 else 'diffuse — property information '
                 'appears distributed rather than localised'})""")

    # ---- persist the operating direction --------------------------------
    store = DirectionStore(os.path.join(args.artifacts, "directions"))
    proj = next((r["projection_spearman"] for r in report["projection_checks"]
                 if r["estimator"] == args.estimator), float("nan"))
    art = DirectionArtifact(
        vector=directions[best["layer"], best["position"]].numpy(),
        property=args.property, layer=best["layer"], site=site,
        position=best["position"], estimator=args.estimator,
        corpus=os.path.basename(args.data), split="train",
        n_samples=report["gate_a_stats"].get("n_positive", 0),
        seed=args.seed, projection_spearman=proj,
        heavy_mean_abs_diff=report["gate_a_stats"].get("heavy_mean_abs_diff", float("nan")),
        model_ckpt=args.ckpt, model_step=mstep,
        extra={"alpha_ref": args.alpha_ref, "delta_property": best["delta_property"]})
    aid = store.save(art)
    print(f"[select] stored direction {aid}  usable={art.is_usable()}")

    with open(os.path.join(args.artifacts, f"selected_{args.property}.json"), "w") as f:
        json.dump({"direction_id": aid, "site": site, "estimator": args.estimator,
                   "baseline_property": baseline_prop,
                   "baseline_heavy": baseline_heavy, **best}, f, indent=2)
    print("\n-> next: scripts/p2_alpha_sweep.py (E2, the headline Pareto frontier)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())