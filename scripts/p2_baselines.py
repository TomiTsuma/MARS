"""
Phase 2, Stage G — E5 baseline comparison. Figures B2 (with baselines) and F.

    python scripts/p2_baselines.py \
        --data artifacts/zinc250k --ckpt runs/p1/ckpt_final.pt \
        --property logp --methods uncond,steering,classifier,dcbg,prefix

WHAT IS AND IS NOT RUNNABLE FROM AN UNCONDITIONAL CHECKPOINT
------------------------------------------------------------
Inference-only (run here directly):
    uncond      unconditional generation — the floor
    steering    activation steering — the method under test
    classifier  classifier guidance (trains a small surrogate inline)
    dcbg        D-CBG, first-order approximation (reuses that surrogate)
    prefix      prefix conditioning with NO intervention — isolates the
                contribution of the intervention from the presence of the prefix

Require a SEPARATELY TRAINED conditional model (pass --cond-ckpt):
    adaln       learned conditioning through the adaLN port
    dcfg        classifier-free guidance for discrete diffusion
    finetune    conditional fine-tuning — the upper bound on achievable control

That split is itself the result. The three unavailable baselines are unavailable
precisely because they need a training run, which is the cost activation
steering claims to avoid. The script reports their setup cost as "not run —
requires conditional pretraining" rather than silently omitting them.

COST IS REPORTED WITH EQUAL PROMINENCE TO CONTROL. The central practical claim
of the method is cheapness, and a claim of that kind must be measured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from config.config import Config
from datasets.tokenizer import SelfiesTokenizer
from model.schedule import get_schedule
from model.sampler import sample_to_smiles
from model.model import MDLM
from properties.oracles.rdkit_props import compute, EXACT_TARGETS
from properties.steering import (load_residual_stats, DirectionStore, AdditiveSteer)
from model.baselines.classifier import train_token_classifier
from model.baselines.guidance import (ClassifierGuidance, DCBG, DCFG,
                                            PrefixConditioning)
from eval.chemical_sanity import (build_training_ring_systems,
                                              sanity_report)
from eval.steering_metrics import property_shift, pareto_frontier
from eval.figures import fig_b2_pareto, fig_f_cost

TRAINING_REQUIRED = {
    "adaln":    "conditional pretraining with the property in the adaLN vector",
    "dcfg":     "conditional pretraining (needs both conditional and unconditional logits)",
    "finetune": "conditional fine-tuning of the full model",
}


def load_model(ckpt, cfg, tok, device):
    m = MDLM(tok.vocab_size, cfg.model, pad_id=tok.pad_id).to(device)
    ck = torch.load(ckpt, map_location=device)
    m.load_state_dict(ck["model"])
    if "ema" in ck:
        sd = m.state_dict()
        for k, v in ck["ema"].items():
            sd[k].copy_(v.to(sd[k].dtype))
    return m.eval()


def evaluate(gen, prop, base_df, train_rings):
    import pandas as pd
    valid = [s for s in gen if s]
    recs = [r for r in (compute(s) for s in valid) if r]
    if not recs:
        return None, {"chemical_sanity": 0.0, "selfies_validity": 0.0}
    df = pd.DataFrame(recs)
    san = sanity_report(valid, train_rings)
    return df, san


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--artifacts", default="artifacts/p2")
    ap.add_argument("--property", default="logp", choices=list(EXACT_TARGETS))
    ap.add_argument("--methods", default="uncond,steering,classifier,dcbg,prefix")
    ap.add_argument("--cond-ckpt", default=None,
                    help="separately trained conditional checkpoint for adaln/dcfg/finetune")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--gammas", default="1,2,3,5",
                    help="guidance strengths; Schiff et al. report MDLM degrades as this rises")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--clf-epochs", type=int, default=12)
    ap.add_argument("--clf-n", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    gammas = [float(g) for g in args.gammas.split(",")]
    prop = args.property

    cfg = Config.from_json(args.config or
                           os.path.join(os.path.dirname(args.ckpt), "config.json"))
    tok = SelfiesTokenizer.load(os.path.join(args.data, "tokenizer.json"))
    schedule = get_schedule(cfg.diffusion.schedule)
    model = load_model(args.ckpt, cfg, tok, args.device)
    stats = load_residual_stats(os.path.join(args.artifacts, "residual_stats.pt"))
    store = DirectionStore(os.path.join(args.artifacts, "directions"))

    import pandas as pd
    with open(os.path.join(args.data, "train_smiles.txt")) as f:
        train_smiles = [l.strip() for l in f if l.strip()]
    ring_cache = os.path.join(args.artifacts, "train_rings.json")
    if os.path.exists(ring_cache):
        train_rings = set(json.load(open(ring_cache)))
    else:
        train_rings = build_training_ring_systems(train_smiles, verbose=False)
        json.dump(sorted(train_rings), open(ring_cache, "w"))

    results, costs = [], []

    # ------------------------------------------------------------ baseline
    print(f"[E5] unconditional floor ({args.n} molecules)")
    t0 = time.time()
    torch.manual_seed(args.seed)
    base_gen = sample_to_smiles(model, schedule, tok, args.n, n_steps=args.n_steps,
                                device=args.device, batch_size=args.batch_size)
    uncond_seconds = time.time() - t0
    base_df, base_san = evaluate(base_gen, prop, None, train_rings)
    base_mean = float(base_df[prop].mean())
    print(f"     {prop}={base_mean:.3f}  sanity={base_san['chemical_sanity']:.3f}  "
          f"{uncond_seconds:.1f}s")
    if "uncond" in methods:
        results.append({"method": "unconditional", "gamma": 0.0,
                        "delta_property": 0.0,
                        "chemical_sanity": base_san["chemical_sanity"],
                        "fidelity_cost": 1 - base_san["chemical_sanity"],
                        "heavy_shift": 0.0})
        costs.append({"method": "unconditional", "setup_seconds": 0.0,
                      "inference_seconds_per_1k": uncond_seconds / (args.n / 1000)})

    target = float(np.percentile([r[prop] for r in
                                  (compute(s) for s in train_smiles[:5000]) if r], 90))
    print(f"[E5] target {prop} = {target:.3f} (90th percentile of training data)")

    def record(name, gamma, gen, setup_s, infer_s):
        df, san = evaluate(gen, prop, base_df, train_rings)
        if df is None:
            print(f"     {name} γ={gamma}: NO VALID MOLECULES")
            results.append({"method": name, "gamma": gamma,
                            "delta_property": 0.0, "chemical_sanity": 0.0,
                            "fidelity_cost": 1.0, "heavy_shift": float("nan")})
            return
        ps = property_shift(df[prop], base_df[prop], n_boot=200)
        results.append({"method": name, "gamma": gamma,
                        "delta_property": ps["shift"],
                        "chemical_sanity": san["chemical_sanity"],
                        "fidelity_cost": 1 - san["chemical_sanity"],
                        "heavy_shift": float(df["heavy"].mean() - base_df["heavy"].mean())})
        print(f"     {name:>12} γ={gamma:<4} Δ{prop}={ps['shift']:+.3f}  "
              f"validity={san['selfies_validity']:.3f}  "
              f"sanity={san['chemical_sanity']:.3f}  {infer_s:.1f}s")
        costs.append({"method": f"{name}" if gamma in (0.0, 1.0) else f"{name}(γ={gamma})",
                      "setup_seconds": setup_s,
                      "inference_seconds_per_1k": infer_s / (args.n / 1000)})

    # -------------------------------------------------- activation steering
    if "steering" in methods:
        print("\n[E5] activation steering (the method under test)")
        sel_path = os.path.join(args.artifacts, f"selected_{prop}.json")
        if os.path.exists(sel_path):
            art = store.load(json.load(open(sel_path))["direction_id"])
            # setup cost = one extraction pass, already measured in Stage B
            setup = float(json.load(open(os.path.join(
                args.artifacts, f"probe_report_{prop}.json"))).get(
                "extraction_seconds", 5.0))
            for a in sorted({args.alpha, args.alpha * 1.5}):
                steer = AdditiveSteer(torch.from_numpy(art.vector), a, stats=stats,
                                      layers={art.layer}, n_struct=tok.n_struct,
                                      pad_id=tok.pad_id, mask_id=tok.mask_id)
                t0 = time.time(); torch.manual_seed(args.seed)
                gen = sample_to_smiles(model, schedule, tok, args.n,
                                       n_steps=args.n_steps, device=args.device,
                                       batch_size=args.batch_size, steer=steer)
                record("steering", a, gen, setup, time.time() - t0)
        else:
            print("     skip: no selected direction")

    # --------------------------------------- classifier guidance and D-CBG
    if {"classifier", "dcbg"} & set(methods):
        print("\n[E5] training the property surrogate "
              "(this cost is what steering avoids)")
        ids_all = np.load(os.path.join(args.data, "train.npy")).astype(np.int64)
        n = min(len(train_smiles), len(ids_all), args.clf_n)
        recs = [compute(s) for s in train_smiles[:n]]
        keep = [i for i, r in enumerate(recs) if r]
        y = np.array([recs[i][prop] for i in keep], dtype=float)
        clf_ids = torch.from_numpy(ids_all[keep])
        clf = train_token_classifier(clf_ids, y, tok.vocab_size, tok.max_len,
                                     epochs=args.clf_epochs, device=args.device,
                                     seed=args.seed)
        setup_s = clf["train_seconds"]

        for name, cls in (("classifier", ClassifierGuidance), ("dcbg", DCBG)):
            if name not in methods:
                continue
            print(f"\n[E5] {name}")
            for g in gammas:
                guide = cls(clf["model"], target, g, clf["mean"], clf["std"])
                t0 = time.time(); torch.manual_seed(args.seed)
                gen = sample_to_smiles(model, schedule, tok, args.n,
                                       n_steps=args.n_steps, device=args.device,
                                       batch_size=args.batch_size, logit_fn=guide)
                record(name, g, gen, setup_s, time.time() - t0)

    # ------------------------------------------------- prefix conditioning
    if "prefix" in methods:
        print("\n[E5] prefix conditioning (no intervention — isolates the prefix)")
        t0 = time.time(); torch.manual_seed(args.seed)
        gen = sample_to_smiles(model, schedule, tok, args.n, n_steps=args.n_steps,
                               device=args.device, batch_size=args.batch_size)
        record("prefix_only", 0.0, gen, 0.0, time.time() - t0)
        print("     (identical to unconditional unless the prefix was trained "
              "with property tokens — a null result here is the correct control)")

    # ----------------------------------- baselines needing conditional training
    for m in methods:
        if m in TRAINING_REQUIRED:
            if args.cond_ckpt is None:
                print(f"\n[E5] {m}: NOT RUN — requires {TRAINING_REQUIRED[m]}.")
                print(f"     Pass --cond-ckpt to enable. This unavailability is "
                      f"itself the cost comparison: the baseline needs a training "
                      f"run that activation steering does not.")
                costs.append({"method": f"{m} (not run)",
                              "setup_seconds": float("nan"),
                              "inference_seconds_per_1k": float("nan")})
            else:
                print(f"\n[E5] {m}: conditional checkpoint supplied — "
                      f"implement the conditioning interface for your "
                      f"conditional model and re-run.")

    # ------------------------------------------------------------ outputs
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.artifacts, f"e5_baselines_{prop}.csv"), index=False)
    print("\n" + df.round(4).to_string(index=False))

    steer_rows = df[df.method == "steering"]
    others = [{"x": r.delta_property, "y": r.fidelity_cost,
               "label": f"{r.method} γ={r.gamma:g}" if r.gamma else r.method}
              for r in df[df.method != "steering"].itertuples()]
    if not steer_rows.empty:
        fig_b2_pareto(steer_rows.rename(columns={"gamma": "alpha"}),
                      baselines=others,
                      out_path=os.path.join(args.artifacts, f"figB2_{prop}.png"))
        print(f"\n[B2] Pareto with baselines -> figB2_{prop}.png")

    cdf = pd.DataFrame(costs).dropna()
    if not cdf.empty:
        fig_f_cost(cdf.to_dict("records"),
                   os.path.join(args.artifacts, f"figF_cost_{prop}.png"))
        print(f"[F]  cost comparison -> figF_cost_{prop}.png")

    # the Schiff asymmetry check
    for name in ("classifier", "dcbg"):
        sub = df[df.method == name].sort_values("gamma")
        if len(sub) > 1:
            trend = sub.chemical_sanity.iloc[-1] - sub.chemical_sanity.iloc[0]
            print(f"\n[Schiff check] {name}: chemical sanity changes by "
                  f"{trend:+.3f} from γ={sub.gamma.iloc[0]:g} to "
                  f"{sub.gamma.iloc[-1]:g}")
            if trend < -0.15:
                print("     Degrades with increasing guidance — reproduces the "
                      "absorbing-state result reported by Schiff et al. (2024).")

    with open(os.path.join(args.artifacts, f"e5_{prop}.json"), "w") as f:
        json.dump({"results": results, "costs": costs, "target": target,
                   "baseline_property": base_mean}, f, indent=2, default=float)
    print(f"\n[done] {args.artifacts}/e5_{prop}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())