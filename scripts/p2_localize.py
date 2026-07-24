"""
Phase 2, Stage F — E3 temporal and depth localisation. Figures D1–D3.

    python scripts/p2_localize.py \
        --data artifacts/zinc250k --ckpt runs/p1/ckpt_final.pt \
        --property logp --axis all

THE THREE-WAY CONFOUND
----------------------
Early-step dominance, if observed, admits three explanations and this script is
built to separate them:

  1. REPRESENTATIONAL  early-layer representations are more malleable.
  2. MECHANICAL        more positions are still maskable early, so steering has
                       greater causal reach.
  3. GENERATIVE-PROCESS  early steps fix composition and late steps only refine,
                       so late intervention has less left to change.

Two controls are applied:

  * FIXED UNMASKING (--fixed-unmasking, default on). Pre-drawn reveal noise is
    shared across every condition, so all conditions unmask the SAME positions
    at the SAME steps. Removes sampling-trajectory variance as an explanation.

  * INTERVENTION-MASS NORMALISATION. Every condition records how many
    (batch x position) elements it actually modified. Reporting efficacy per
    unit mass separates (2) from (1) and (3): if FIRST-k only wins because it
    touches more positions, the mass-normalised curves converge.

Separating (1) from (3) needs the property-resolved design — pass several
properties with --properties to produce Figure D3.
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
from model.sampler import sample_to_smiles, build_reveal_noise
from model.model import MDLM
from properties.oracles.rdkit_props import compute, EXACT_TARGETS
from properties.steering import (load_residual_stats, DirectionStore, AdditiveSteer,
                                  schedule_from_spec, position_from_spec,
                                  StepWindow, SWEEP_SPECS)
from eval.chemical_sanity import sanity_report
from eval.figures import fig_d1_surface, fig_d2_schedules, fig_d3_timing


def load_model(ckpt, cfg, tok, device):
    m = MDLM(tok.vocab_size, cfg.model, pad_id=tok.pad_id).to(device)
    ck = torch.load(ckpt, map_location=device)
    m.load_state_dict(ck["model"])
    if "ema" in ck:
        sd = m.state_dict()
        for k, v in ck["ema"].items():
            sd[k].copy_(v.to(sd[k].dtype))
    return m.eval()


def run_condition(model, schedule, tok, art, stats, alpha, prop, args,
                  layers=None, sched_spec="all", pos_spec="all",
                  reveal=None, device="cpu"):
    """One steering condition -> (delta property, sanity, intervention mass)."""
    steer = AdditiveSteer(
        torch.from_numpy(art.vector), alpha, stats=stats,
        layers=layers if layers is not None else {art.layer},
        schedule=schedule_from_spec(sched_spec),
        positions=position_from_spec(pos_spec),
        n_struct=tok.n_struct, pad_id=tok.pad_id, mask_id=tok.mask_id)
    torch.manual_seed(args.seed)
    gen = sample_to_smiles(model, schedule, tok, args.n, n_steps=args.n_steps,
                           device=device, batch_size=args.batch_size,
                           steer=steer, reveal_noise=reveal)
    valid = [s for s in gen if s]
    recs = [r for r in (compute(s) for s in valid) if r]
    if not recs:
        return {"prop_mean": float("nan"), "heavy_mean": float("nan"),
                "sanity": 0.0, "mass": steer.mass, "n_valid": 0}
    return {"prop_mean": float(np.nanmean([r[prop] for r in recs])),
            "heavy_mean": float(np.nanmean([r["heavy"] for r in recs])),
            "sanity": sanity_report(valid)["chemical_sanity"],
            "mass": steer.mass, "n_valid": len(valid)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--artifacts", default="artifacts/p2")
    ap.add_argument("--property", default="logp", choices=list(EXACT_TARGETS))
    ap.add_argument("--properties", default=None,
                    help="comma list for the D3 property-timing experiment (E6)")
    ap.add_argument("--axis", default="all",
                    choices=["all", "when", "depth", "position", "surface", "timing"])
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--kappas", default="1,2,4,8,16,32")
    ap.add_argument("--n-windows", type=int, default=8,
                    help="step windows for the D1 surface and D3 timing scan")
    ap.add_argument("--fixed-unmasking", action="store_true", default=True)
    ap.add_argument("--no-fixed-unmasking", dest="fixed_unmasking",
                    action="store_false")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.from_json(args.config or
                           os.path.join(os.path.dirname(args.ckpt), "config.json"))
    tok = SelfiesTokenizer.load(os.path.join(args.data, "tokenizer.json"))
    schedule = get_schedule(cfg.diffusion.schedule)
    model = load_model(args.ckpt, cfg, tok, args.device)
    stats = load_residual_stats(os.path.join(args.artifacts, "residual_stats.pt"))
    store = DirectionStore(os.path.join(args.artifacts, "directions"))

    with open(os.path.join(args.artifacts, f"selected_{args.property}.json")) as f:
        sel = json.load(f)
    art = store.load(sel["direction_id"])
    base_prop = sel["baseline_property"]
    n_layers = len(model.blocks) + 1
    kappas = [int(k) for k in args.kappas.split(",") if int(k) <= args.n_steps]

    reveal = (build_reveal_noise(args.n_steps, args.batch_size, tok.max_len,
                                 seed=args.seed) if args.fixed_unmasking else None)
    print(f"[E3] direction {art.id} layer {art.layer} | alpha={args.alpha} | "
          f"N={args.n_steps} | fixed-unmasking={args.fixed_unmasking}")
    if not args.fixed_unmasking:
        print("     WARNING: without the fixed-unmasking control, early-step "
              "dominance cannot be separated from sampling-trajectory variance.")

    results = {"property": args.property, "alpha": args.alpha,
               "fixed_unmasking": args.fixed_unmasking, "baseline": base_prop}
    do = lambda k: args.axis in ("all", k)

    # ---------------------------------------------------- D2 schedule ablation
    if do("when"):
        print("\n[D2] schedule ablation")
        full = run_condition(model, schedule, tok, art, stats, args.alpha,
                             args.property, args, sched_spec="all",
                             reveal=reveal, device=args.device)
        curves = {}
        for fam in ("first", "last", "every"):
            vals, mass, ks = [], [], []
            for k in kappas:
                spec = (f"every:{max(1, args.n_steps // k)}" if fam == "every"
                        else f"{fam}:{k}")
                r = run_condition(model, schedule, tok, art, stats, args.alpha,
                                  args.property, args, sched_spec=spec,
                                  reveal=reveal, device=args.device)
                vals.append(abs(r["prop_mean"] - base_prop))
                mass.append(r["mass"]); ks.append(k)
                print(f"     {spec:12} Δ={vals[-1]:+.3f}  mass={r['mass']:,}  "
                      f"sanity={r['sanity']:.3f}")
            curves[f"{fam}_k"] = {"kappa": ks, "values": vals, "mass": mass}
        results["schedules"] = curves
        results["full_schedule"] = abs(full["prop_mean"] - base_prop)
        fig_d2_schedules(curves, results["full_schedule"],
                         os.path.join(args.artifacts, f"figD2_{args.property}.png"))
        fig_d2_schedules(curves, None,
                         os.path.join(args.artifacts,
                                      f"figD2_{args.property}_massnorm.png"),
                         normalise_by_mass=True)
        print(f"     -> Figure D2 (raw and mass-normalised) written")
        f_k = np.array(curves["first_k"]["values"])
        l_k = np.array(curves["last_k"]["values"])
        results["early_dominance_ratio"] = float(np.nanmean(f_k) /
                                                 max(np.nanmean(l_k), 1e-8))
        print(f"     early/late efficacy ratio = "
              f"{results['early_dominance_ratio']:.2f}"
              f"  ({'early dominates' if results['early_dominance_ratio'] > 1.5 else 'no strong asymmetry'})")

    # ------------------------------------------------------- depth ablation
    if do("depth"):
        print("\n[depth] layer-band ablation")
        third = max(1, n_layers // 3)
        bands = {"early": set(range(0, third)),
                 "middle": set(range(third, 2 * third)),
                 "late": set(range(2 * third, n_layers)),
                 "all": set(range(n_layers)),
                 f"source_only(L{art.layer})": {art.layer}}
        depth = {}
        for name, ls in bands.items():
            r = run_condition(model, schedule, tok, art, stats, args.alpha,
                              args.property, args, layers=ls,
                              reveal=reveal, device=args.device)
            depth[name] = {"delta": abs(r["prop_mean"] - base_prop),
                           "sanity": r["sanity"], "mass": r["mass"]}
            print(f"     {name:>18}: Δ={depth[name]['delta']:+.3f}  "
                  f"sanity={r['sanity']:.3f}")
        results["depth"] = depth

    # ---------------------------------------------------- position ablation
    if do("position"):
        print("\n[position] token-position ablation")
        pos = {}
        for spec in SWEEP_SPECS:
            r = run_condition(model, schedule, tok, art, stats, args.alpha,
                              args.property, args, pos_spec=spec,
                              reveal=reveal, device=args.device)
            pos[spec] = {"delta": abs(r["prop_mean"] - base_prop),
                         "sanity": r["sanity"], "mass": r["mass"]}
            print(f"     {spec:>8}: Δ={pos[spec]['delta']:+.3f}  "
                  f"mass={r['mass']:,}  sanity={r['sanity']:.3f}")
        results["positions"] = pos
        if "masked" in pos and "frozen" in pos:
            print(f"     masked vs frozen: {pos['masked']['delta']:.3f} vs "
                  f"{pos['frozen']['delta']:.3f}  — frozen positions can only "
                  f"act indirectly through attention")

    # ------------------------------------------------- D1 layer x step surface
    if do("surface"):
        print("\n[D1] layer × timestep surface")
        edges = np.linspace(0, args.n_steps, args.n_windows + 1).astype(int)
        third = max(1, n_layers // 3)
        bands = [("early", set(range(0, third))),
                 ("middle", set(range(third, 2 * third))),
                 ("late", set(range(2 * third, n_layers)))]
        Z = np.zeros((len(bands), args.n_windows))
        for bi, (_, ls) in enumerate(bands):
            for wi in range(args.n_windows):
                r = run_condition(model, schedule, tok, art, stats, args.alpha,
                                  args.property, args, layers=ls,
                                  sched_spec=f"window:{edges[wi]}:{edges[wi+1]}",
                                  reveal=reveal, device=args.device)
                Z[bi, wi] = abs(r["prop_mean"] - base_prop)
            print(f"     band {bands[bi][0]:>6}: "
                  f"{[round(float(z),3) for z in Z[bi]]}")
        results["surface"] = {"Z": Z.tolist(),
                              "bands": [b[0] for b in bands],
                              "step_edges": edges.tolist()}
        fig_d1_surface(Z, layer_labels=[b[0] for b in bands],
                       step_labels=[f"{edges[i]}-{edges[i+1]}"
                                    for i in range(args.n_windows)],
                       out_path=os.path.join(args.artifacts,
                                             f"figD1_{args.property}.png"))
        print("     -> Figure D1 written")

    # ------------------------------------------- D3 property-timing (E6)
    if do("timing") and args.properties:
        print("\n[D3] property–timing correspondence (E6)")
        edges = np.linspace(0, args.n_steps, args.n_windows + 1).astype(int)
        rows = []
        for prop in [p.strip() for p in args.properties.split(",")]:
            sel_path = os.path.join(args.artifacts, f"selected_{prop}.json")
            if not os.path.exists(sel_path):
                print(f"     skip {prop}: no selected direction")
                continue
            with open(sel_path) as f:
                s2 = json.load(f)
            a2 = store.load(s2["direction_id"])
            curve = []
            for wi in range(args.n_windows):
                r = run_condition(model, schedule, tok, a2, stats, args.alpha,
                                  prop, args,
                                  sched_spec=f"window:{edges[wi]}:{edges[wi+1]}",
                                  reveal=reveal, device=args.device)
                curve.append(abs(r["prop_mean"] - s2["baseline_property"]))
            c = np.asarray(curve, float)
            if not np.isfinite(c).any() or c.max() <= 0:
                continue
            w = c / c.sum()
            centres = (edges[:-1] + edges[1:]) / 2 / args.n_steps
            centre = float((w * centres).sum())
            width = float(2 * np.sqrt(((centres - centre) ** 2 * w).sum()))
            rows.append({"label": prop, "centre": centre,
                         "width": max(width, .05), "curve": curve})
            print(f"     {prop:>8}: critical window centre = {centre:.3f}")
        if rows:
            results["timing"] = rows
            fig_d3_timing(rows, os.path.join(args.artifacts, "figD3_timing.png"))
            order = [r["label"] for r in sorted(rows, key=lambda r: r["centre"])]
            print(f"     ordering (earliest → latest): {' < '.join(order)}")
            print("     Prediction: window position should track the structural "
                  "level the property depends on.")

    out = os.path.join(args.artifacts, f"e3_localize_{args.property}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())