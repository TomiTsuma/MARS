"""
Phase 2, Stage F — E4 specificity and confound control. Figures C1, C2.

    python scripts/p2_specificity.py \
        --data artifacts/zinc250k --ckpt runs/p1/ckpt_final.pt \
        --properties logp,qed,tpsa,sa,rings

WHY THIS EXPERIMENT CAN SINK THE PHASE
--------------------------------------
Most molecular properties correlate with molecular size. A direction that
simply makes molecules bigger will improve MAE on LogP, MW and several ADMET
endpoints at once, and will look exactly like property control. Three defences
run here, together:

  1. The off-target matrix, which reveals whether "different" property
     directions are one latent factor under several names.
  2. Heavy-atom shift reported for EVERY row — the matrix cannot distinguish a
     property axis from a size axis without it.
  3. Atom-count-stratified shift: if the effect vanishes once size is held
     fixed, the direction encodes size and the headline result is not what it
     appears to be.

Two further analyses characterise the mechanism rather than the effect:

  * Effective dimensionality (Figure C2) tests the approximate
    one-dimensionality claim inherited from the text precedent instead of
    assuming it.
  * Fragment-frequency shift detects steering-by-substitution — a legitimate
    mechanism and an honest finding, but a different claim from the one the
    phase sets out to make.
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
from properties.oracles.rdkit_props import compute, EXACT_TARGETS, CONFOUND_CONTROLS
from properties.steering import (load_residual_stats, DirectionStore, AdditiveSteer,
                                  collect_activations, PrefixReducer, PooledReducer,
                                  MaskReducer, effective_dim, per_decile_directions,
                                  offtarget_matrix, fragment_frequency_shift,
                                  direction_similarity)
from eval.chemical_sanity import sanity_report
from eval.steering_metrics import atom_count_stratified_shift
from eval.figures import fig_c1_offtarget, fig_c2_scree


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--artifacts", default="artifacts/p2")
    ap.add_argument("--properties", default="logp,qed,tpsa,sa,rings")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--probe-n", type=int, default=512,
                    help="molecules for the dimensionality / curvature analysis")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    props = [p.strip() for p in args.properties.split(",")]
    measured = list(dict.fromkeys(props + list(CONFOUND_CONTROLS)))

    cfg = Config.from_json(args.config or
                           os.path.join(os.path.dirname(args.ckpt), "config.json"))
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
    stats = load_residual_stats(os.path.join(args.artifacts, "residual_stats.pt"))
    store = DirectionStore(os.path.join(args.artifacts, "directions"))

    import pandas as pd

    # ---------------------------------------------------------- baseline
    print(f"[E4] unconditional baseline ({args.n} molecules)")
    torch.manual_seed(args.seed)
    base = sample_to_smiles(model, schedule, tok, args.n, n_steps=args.n_steps,
                            device=args.device, batch_size=args.batch_size)
    base_recs = [r for r in (compute(s) for s in base if s) if r]
    base_df = pd.DataFrame(base_recs)
    print("     " + "  ".join(f"{p}={base_df[p].mean():.3f}"
                              for p in measured if p in base_df))

    # ------------------------------------------------- steer each property
    rows, heavy_shift, dirs = [], [], {}
    for prop in props:
        sel_path = os.path.join(args.artifacts, f"selected_{prop}.json")
        if not os.path.exists(sel_path):
            print(f"[E4] skip {prop}: no selected direction "
                  f"(run p2_sweep_select.py --property {prop})")
            continue
        with open(sel_path) as f:
            sel = json.load(f)
        art = store.load(sel["direction_id"])
        dirs[prop] = torch.from_numpy(art.vector)

        steer = AdditiveSteer(torch.from_numpy(art.vector), args.alpha, stats=stats,
                              layers={art.layer}, n_struct=tok.n_struct,
                              pad_id=tok.pad_id, mask_id=tok.mask_id)
        torch.manual_seed(args.seed)
        gen = sample_to_smiles(model, schedule, tok, args.n, n_steps=args.n_steps,
                               device=args.device, batch_size=args.batch_size,
                               steer=steer)
        valid = [s for s in gen if s]
        recs = [r for r in (compute(s) for s in valid) if r]
        gdf = pd.DataFrame(recs)
        gdf["steered_property"] = prop
        gdf["alpha"] = args.alpha
        rows.append(gdf)

        hs = float(gdf["heavy"].mean() - base_df["heavy"].mean())
        heavy_shift.append(hs)
        san = sanity_report(valid)
        print(f"[E4] steered {prop:>6}: Δ{prop}="
              f"{gdf[prop].mean()-base_df[prop].mean():+.3f}  "
              f"Δheavy={hs:+.2f}  sanity={san['chemical_sanity']:.3f}")

    if not rows:
        print("[E4] no directions available — nothing to do")
        return 1
    gen_df = pd.concat(rows, ignore_index=True)
    gen_df.to_parquet(os.path.join(args.artifacts, "specificity_generations.parquet"),
                      index=False)

    # ------------------------------------------------------ C1 off-target
    print("\n[C1] off-target matrix (normalised by unconditional σ)")
    om = offtarget_matrix(gen_df, props, base_df, alpha_ref=args.alpha)
    M = om["matrix"]
    print(M[[p for p in props if p in M.columns]].round(3).to_string())
    print(f"     diagonal mean = {om['diagonal_mean']:.3f}  "
          f"off-diagonal mean = {om['offdiagonal_mean']:.3f}  "
          f"dominance = {om['diagonal_dominance']:.2f}")
    if om["diagonal_dominance"] < 1.5:
        print("     WARNING: weak diagonal dominance. The directions may be "
              "steering a shared latent factor (most likely molecular size) "
              "under several names — read the heavy-atom row below.")
    print(f"     heavy-atom shift per steered property: "
          f"{dict(zip(props[:len(heavy_shift)], [round(h,2) for h in heavy_shift]))}")
    fig_c1_offtarget(M, os.path.join(args.artifacts, "figC1_offtarget.png"),
                     heavy=heavy_shift)

    # ------------------------------------- atom-count-stratified confound check
    print("\n[confound] atom-count-stratified shift")
    strat = {}
    for prop in props:
        sub = gen_df[gen_df.steered_property == prop]
        if sub.empty:
            continue
        s = atom_count_stratified_shift(sub, base_df, prop)
        strat[prop] = s
        raw = float(sub[prop].mean() - base_df[prop].mean())
        within = s["mean_within_stratum_shift"]
        retained = (within / raw) if raw else float("nan")
        flag = ("SIZE-CONFOUNDED" if np.isfinite(retained) and retained < 0.5
                else "survives")
        print(f"     {prop:>6}: raw Δ={raw:+.3f}  within-stratum Δ={within:+.3f}  "
              f"retained={retained:.2f}  [{flag}]")

    # ------------------------------------------------ C2 effective dimensionality
    print("\n[C2] effective dimensionality")
    with open(os.path.join(args.data, "train_smiles.txt")) as f:
        train_smiles = [l.strip() for l in f if l.strip()]
    ids_all = np.load(os.path.join(args.data, "train.npy")).astype(np.int64)
    n = min(len(train_smiles), len(ids_all))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=min(args.probe_n, n), replace=False)
    tr_recs = [compute(train_smiles[i]) for i in idx]
    ok = [i for i, r in enumerate(tr_recs) if r]
    idx, tr_recs = idx[ok], [tr_recs[i] for i in ok]
    probe_ids = torch.from_numpy(ids_all[idx])

    with open(os.path.join(args.artifacts, f"probe_report_{props[0]}.json")) as f:
        site = json.load(f)["decision"]["primary_site"]
    reducer = {"prefix": PrefixReducer(tok.n_struct), "pooled": PooledReducer(),
               "masked": MaskReducer()}[site]
    H = collect_activations(model, probe_ids, tok, schedule, reducer,
                            t=0.5 if site == "masked" else 0.0,
                            batch_size=args.batch_size, device=args.device)

    scree, curvature = {}, {}
    for prop in props:
        sel_path = os.path.join(args.artifacts, f"selected_{prop}.json")
        if not os.path.exists(sel_path):
            continue
        art = store.load(json.load(open(sel_path))["direction_id"])
        y = np.array([r[prop] for r in tr_recs], dtype=float)
        med = np.median(y)
        hi = H[y > med]; lo = H[y <= med]
        m = min(len(hi), len(lo))
        if m < 10:
            continue
        ed = effective_dim(hi[:m], lo[:m], art.layer, art.position)
        scree[prop] = ed["variance_explained"][:8]
        curvature[prop] = per_decile_directions(H, y, art.layer, art.position)
        print(f"     {prop:>6}: PC1={ed['pc1']:.3f}  "
              f"n_for_90%={ed['n_for_90pct']}  "
              f"{'approx 1-D' if ed['approximately_1d'] else 'NOT 1-D'}  |  "
              f"min inter-decile cos={curvature[prop]['min_offdiag_cosine']:+.3f}"
              f"{'  CURVED MANIFOLD' if curvature[prop]['curved_manifold'] else ''}")
    if scree:
        fig_c2_scree(scree, os.path.join(args.artifacts, "figC2_scree.png"))

    # --------------------------------------------- direction cross-similarity
    if len(dirs) > 1:
        sim = direction_similarity(dirs)
        print("\n[similarity] pairwise cosine between property directions")
        print(pd.DataFrame(sim["matrix"], index=sim["names"],
                           columns=sim["names"]).round(3).to_string())
        print("     High similarity here plus a dense off-target matrix is "
              "strong evidence for a single shared latent factor.")

    # -------------------------------------------------- fragment substitution
    print("\n[fragments] substitution check")
    frag = {}
    for prop in props:
        sub = gen_df[gen_df.steered_property == prop]
        if sub.empty:
            continue
        merged = pd.concat([base_df.assign(alpha=0.0), sub], ignore_index=True)
        try:
            fr = fragment_frequency_shift(merged)
            frag[prop] = {"top_k_positive_share": fr["top_k_positive_share"],
                          "substitution_dominated": fr["substitution_dominated"]}
            print(f"     {prop:>6}: top-25 fragment share of positive shift = "
                  f"{fr['top_k_positive_share']:.3f}"
                  f"{'   SUBSTITUTION-DOMINATED' if fr['substitution_dominated'] else ''}")
            fr["table"].head(40).to_csv(
                os.path.join(args.artifacts, f"fragments_{prop}.csv"), index=False)
        except Exception as e:
            print(f"     {prop:>6}: fragment analysis unavailable ({e})")

    out = {"alpha": args.alpha, "properties": props,
           "offtarget": {"matrix": M.to_dict(),
                         "diagonal_mean": om["diagonal_mean"],
                         "offdiagonal_mean": om["offdiagonal_mean"],
                         "diagonal_dominance": om["diagonal_dominance"]},
           "heavy_shift": dict(zip(props[:len(heavy_shift)], heavy_shift)),
           "stratified": {k: v["mean_within_stratum_shift"] for k, v in strat.items()},
           "effective_dim": scree,
           "curvature": {k: {"min_cos": v["min_offdiag_cosine"],
                             "curved": v["curved_manifold"]}
                         for k, v in curvature.items()},
           "fragments": frag}
    with open(os.path.join(args.artifacts, "e4_specificity.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[done] {args.artifacts}/e4_specificity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())