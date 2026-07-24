"""
Phase 2, Stages A and B — the first artefact of the phase.

    python scripts/p2_probe_sites.py \
        --data artifacts/zinc250k \
        --ckpt runs/phase1/ckpt_final.pt \
        --property logp

What this does, in order:

    1. Annotate the training split with exact RDKit oracles (cached).
    2. Build contrastive sets D+ / D- with stratified heavy-atom matching.
    3. GATE A — verify the sets separate on the property and NOT on size.
    4. Collect residual-stream activations for three extraction-site designs.
    5. Ridge-probe each site at each layer against the property (held-out R^2).
    6. GATE B — decide which extraction site becomes primary.
    7. Extract directions and run the held-out projection check.

Why run this first
------------------
It costs a few hundred forward passes, generates no molecules, and answers the
question the whole phase depends on: does the structural prefix installed
during Phase 1 pretraining actually carry property information? If it does
not, the primary extraction-site design changes and the change should happen
BEFORE any generation budget is spent.

Exit code 0 if both gates pass, 1 otherwise.
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
from model.model import MDLM
from model.hooks import verify_noop_identity, ResidualStats
from properties.oracles.rdkit_props import annotate_corpus, EXACT_TARGETS
from properties.steering import (match_atom_counts, gate_a_check,
                                  PrefixReducer, PooledReducer, MaskReducer,
                                  collect_activations, diff_of_means, ridge_probe,
                                  probe_site_designs, interpret_probe,
                                  projection_check, bootstrap_directions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir from gate1_prepare.py")
    ap.add_argument("--ckpt", required=True, help="Phase 1 checkpoint")
    ap.add_argument("--config", default=None, help="defaults to <ckpt dir>/config.json") #runs\phase1\20260723_073440\config.json
    ap.add_argument("--property", default="logp", choices=list(EXACT_TARGETS))
    ap.add_argument("--n-per-side", type=int, default=256)
    ap.add_argument("--n-strata", type=int, default=8)
    ap.add_argument("--probe-n", type=int, default=1024,
                    help="molecules used for the linear probe")
    ap.add_argument("--mask-t", type=float, default=0.5,
                    help="diffusion time for the mask-position site (S3)")
    ap.add_argument("--out", default="artifacts/p2")
    ap.add_argument("--raw", action="store_true", help="use raw weights, not EMA")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---------------------------------------------------------------- setup
    cfg_path = args.config or os.path.join(os.path.dirname(args.ckpt), "config.json")
    cfg = Config.from_json(cfg_path)
    tok = SelfiesTokenizer.load(os.path.join(args.data, "tokenizer.json"))
    schedule = get_schedule(cfg.diffusion.schedule)

    model = MDLM(tok.vocab_size, cfg.model, pad_id=tok.pad_id).to(args.device)
    ck = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ck["model"])
    if not args.raw and "ema" in ck:
        sd = model.state_dict()
        for k, v in ck["ema"].items():
            sd[k].copy_(v.to(sd[k].dtype))
        print("[setup] loaded EMA weights")
    model.eval()

    print(f"[setup] checkpoint step {ck.get('step')}  |  vocab {tok.vocab_size}  "
          f"|  n_struct {tok.n_struct}  |  layers {len(model.blocks)}")

    # prerequisite: instrumentation must be inert
    ids_all = np.load(os.path.join(args.data, "train.npy")).astype(np.int64)
    probe_x = torch.from_numpy(ids_all[:2]).to(args.device)
    if not verify_noop_identity(model, probe_x,
                                torch.full((2,), 0.5, device=args.device)):
        print("[setup] FAIL: hooks are not inert. Fix before proceeding.")
        return 1

    # ------------------------------------------------------- STAGE A: oracles
    print(f"\n{'='*66}\nSTAGE A  —  oracles and contrastive sets\n{'='*66}")
    with open(os.path.join(args.data, "train_smiles.txt")) as f:
        smiles = [l.strip() for l in f if l.strip()]
    n = min(len(smiles), len(ids_all))
    smiles, ids_all = smiles[:n], ids_all[:n]
    print(f"[A] corpus rows aligned: {n:,}")

    props_path = os.path.join(args.out, "props.parquet")
    if os.path.exists(props_path):
        import pandas as pd
        df = pd.read_parquet(props_path)
        print(f"[A] loaded cached annotations: {len(df):,}")
    else:
        print("[A] annotating with exact RDKit oracles ...")
        df = annotate_corpus(smiles, out_path=props_path, n_jobs=8)

    # keep the frame index aligned with the tokenised array
    df = df.reset_index(drop=True)
    keep = min(len(df), len(ids_all))
    df, ids_all = df.iloc[:keep].reset_index(drop=True), ids_all[:keep]

    cset = match_atom_counts(df, args.property,
                             n_per_side=args.n_per_side,
                             n_strata=args.n_strata,
                             seed=args.seed,
                             corpus=os.path.basename(args.data), split="train")
    cset_path = os.path.join(args.out, f"contrastive_{args.property}.json")
    cset.save(cset_path)
    print(f"[A] wrote {cset_path}")

    print()
    gate_a = gate_a_check(cset)
    if not gate_a:
        print("\n[A] GATE A FAILED — extracting from these sets would produce a "
              "molecular-size direction. Stopping.")
        return 1

    # -------------------------------------------------- STAGE B: the probe
    print(f"\n{'='*66}\nSTAGE B  —  extraction-site linear probe\n{'='*66}")

    rng = np.random.default_rng(args.seed)
    probe_idx = rng.choice(len(df), size=min(args.probe_n, len(df)), replace=False)
    probe_ids = torch.from_numpy(ids_all[probe_idx])
    y_probe = df[args.property].to_numpy()[probe_idx]
    print(f"[B] probing on {len(probe_idx):,} molecules")

    reducers = {
        "prefix": PrefixReducer(tok.n_struct),
        "pooled": PooledReducer(),
        "masked": MaskReducer(),
    }
    H_by_site = {}
    for name, red in reducers.items():
        t_val = args.mask_t if name == "masked" else 0.0
        H = collect_activations(model, probe_ids, tok, schedule=schedule,
                                reducer=red, t=t_val,
                                n_realizations=3 if name == "masked" else 1,
                                batch_size=args.batch_size, device=args.device,
                                seed=args.seed)
        H_by_site[name] = H
        print(f"[B]   {name:>8}: activations {tuple(H.shape)}"
              f"{'   (t=%.2f)' % t_val if t_val else ''}")

    results = probe_site_designs(H_by_site, y_probe, seed=args.seed)
    print()
    decision = interpret_probe(results)

    # correctness check: layer 0 prefix must be ~0 (identical across molecules)
    l0 = results["prefix"]["r2_per_layer"][0]
    if l0 > 0.2:
        print(f"\n[B] WARNING: layer-0 prefix R^2 = {l0:.3f}. The prefix tokens are "
              "identical across molecules, so layer 0 should carry no signal. "
              "Suspect a bug in activation collection.")

    # ------------------------------------------- directions on the chosen site
    print(f"\n{'='*66}\nDIRECTIONS  —  site '{decision['primary_site']}'\n{'='*66}")
    site = decision["primary_site"]
    red = reducers[site]
    t_val = args.mask_t if site == "masked" else 0.0

    pos_ids = torch.from_numpy(ids_all[np.array(cset.positive_idx)])
    neg_ids = torch.from_numpy(ids_all[np.array(cset.negative_idx)])
    H_pos = collect_activations(model, pos_ids, tok, schedule, red, t=t_val,
                                batch_size=args.batch_size, device=args.device)
    H_neg = collect_activations(model, neg_ids, tok, schedule, red, t=t_val,
                                batch_size=args.batch_size, device=args.device)

    V_dm = diff_of_means(H_pos, H_neg)
    V_rp = ridge_probe(H_by_site[site], y_probe)
    _, boot_std = bootstrap_directions(H_pos, H_neg, n_boot=50, seed=args.seed)
    print(f"[D] diff-of-means {tuple(V_dm.shape)} | ridge-probe {tuple(V_rp.shape)}")

    best_layer = results[site]["best_layer"]
    n_sites = V_dm.shape[1]
    rows = []
    for s in range(n_sites):
        for est_name, V in (("diffmeans", V_dm), ("ridge", V_rp)):
            rho = projection_check(H_by_site[site], V[best_layer, s],
                                   y_probe, best_layer, s)
            rows.append({"site": site, "estimator": est_name, "layer": best_layer,
                         "position": s, "projection_spearman": rho,
                         "bootstrap_cos_std": float(boot_std[best_layer, s])})
    rows.sort(key=lambda r: -abs(r["projection_spearman"]))
    print(f"[D] held-out projection check at layer {best_layer} "
          f"(want |rho| > 0.3, ideally > 0.5):")
    for r in rows[:6]:
        flag = "OK " if abs(r["projection_spearman"]) > 0.3 else "LOW"
        print(f"      [{flag}] {r['estimator']:>9} pos {r['position']}: "
              f"rho = {r['projection_spearman']:+.3f}  "
              f"boot_cos_std = {r['bootstrap_cos_std']:.3f}")

    # residual stats — the units every alpha is expressed in
    stats = ResidualStats(len(model.blocks) + 1)
    stats.std = torch.stack([H_by_site["pooled"][:, l].float().std()
                             for l in range(len(model.blocks) + 1)])
    torch.save({"std": stats.std}, os.path.join(args.out, "residual_stats.pt"))
    print(f"[D] residual sigma by layer: "
          f"{[round(float(x),3) for x in stats.std[:5]]} ... "
          f"{[round(float(x),3) for x in stats.std[-2:]]}")

    # ------------------------------------------------------------- artefacts
    np.savez(os.path.join(args.out, f"directions_{args.property}_{site}.npz"),
             diffmeans=V_dm.numpy(), ridge=V_rp.numpy(),
             bootstrap_cos_std=boot_std.numpy(), best_layer=best_layer)
    report = {
        "property": args.property, "checkpoint": args.ckpt,
        "gate_a": gate_a, "gate_a_stats": cset.match_stats,
        "probe": results, "decision": decision,
        "projection_checks": rows,
        "residual_sigma": [float(x) for x in stats.std],
    }
    with open(os.path.join(args.out, f"probe_report_{args.property}.json"), "w") as f:
        json.dump(report, f, indent=2)

    best_rho = max(abs(r["projection_spearman"]) for r in rows)
    gate_b = not decision["prefix_inert"] or decision["primary_site"] != "prefix"
    ok = gate_a and gate_b and best_rho > 0.3

    print(f"\n{'='*66}")
    print(f"GATE A (contrastive matching)   : {'PASS' if gate_a else 'FAIL'}")
    print(f"GATE B (site carries signal)    : {'PASS' if gate_b else 'FAIL'}")
    print(f"Projection check (|rho| > 0.3)  : {'PASS' if best_rho > 0.3 else 'FAIL'}"
          f"   (best |rho| = {best_rho:.3f})")
    print(f"-> {'PROCEED to Stage C (sweep and select)' if ok else 'DO NOT PROCEED'}")
    print(f"   artefacts in {args.out}/")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())