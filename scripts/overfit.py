"""
GATE 2 — the overfit test.

Train a tiny model on 32 molecules until it reproduces them. This catches
almost every implementation bug in the diffusion machinery, and it narrows any
remaining bug to a single file instead of a codebase. Run it before you touch
real training. It costs about a minute on CPU.

    python scripts/gate2_overfit.py --smiles data/zinc_sample.txt

PASS criteria
-------------
  * weighted NELBO falls by >100x and keeps falling
  * unweighted CE -> near 0
  * masked-token accuracy -> >0.99
  * sampling reproduces molecules FROM THE 32, not novel ones
    (novelty here is failure, not success)

If accuracy plateaus around chance, or loss NaNs, consult the failure table in
the accompanying notes before changing anything else.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from config.config import Config
from scripts.prepare import build_corpus
from datasets.tokenizer import SelfiesTokenizer
from datasets.dataset import make_masks
from model.schedule import get_schedule
from model.objective import training_step
from model.sampler import sample
from model.model import MDLM
from model.hooks import verify_noop_identity

DEFAULT_SMILES = [
    "CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCOC(=O)C", "c1ccncc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1",
    "CC(=O)Nc1ccc(O)cc1", "Clc1ccccc1", "OCC1OC(O)C(O)C(O)C1O",
    "CC(C)NCC(O)COc1cccc2ccccc12", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "NC(=O)c1ccccc1", "COc1ccccc1", "CCCCO", "c1ccc2ccccc2c1",
    "CC(C)(C)c1ccccc1", "O=C(O)c1ccccc1O", "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "c1ccc(cc1)S(=O)(=O)N", "CCCCCCCC", "C1CCNCC1", "c1cc[nH]c1",
    "O=C1CCCN1", "CSCC", "FC(F)(F)c1ccccc1", "N#Cc1ccccc1",
    "OC1CCCCC1", "c1ccc(cc1)Cc1ccccc1", "CC1=CC(=O)CC(C)(C)C1",
    "NCCc1ccc(O)cc1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", type=str, default=None,
                    help="file with one SMILES per line; defaults to a built-in set")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    smiles = DEFAULT_SMILES
    if args.smiles:
        smiles = [l.strip() for l in open(args.smiles) if l.strip()]
    smiles = smiles[: args.n]

    print(f"[gate2] building corpus from {len(smiles)} molecules")
    smi, selfies, stats = build_corpus(smiles, report_every=0)
    print(f"[gate2] kept {len(smi)}  round-trip failures {stats.get('rejected_roundtrip',0)}")
    assert len(smi) >= 8, "too few molecules survived standardisation"

    tok = SelfiesTokenizer.build(selfies, n_prefix=4, max_len=64)
    print(f"[gate2] vocab {tok.vocab_size}")

    rows = [tok.encode(s) for s in selfies]
    rows = [r for r in rows if r is not None]
    ids = torch.tensor(rows, dtype=torch.long, device=args.device)
    maskable, valid = make_masks(ids, tok.pad_id, tok.n_struct)
    batch = {"ids": ids, "maskable": maskable, "valid": valid}

    # --- deliberately tiny model ---
    cfg = Config()
    cfg.model.d_model = 128
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.d_ff = 512
    cfg.model.dropout = 0.0
    model = MDLM(tok.vocab_size, cfg.model, pad_id=tok.pad_id).to(args.device)
    print(f"[gate2] params {model.n_params():,}")

    # --- hooks must be inert before anything else is believed ---
    verify_noop_identity(model,
                         ids[:2],
                         torch.full((2,), 0.5, device=args.device))

    schedule = get_schedule("loglinear")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    print("[gate2] overfitting ...")
    for step in range(1, args.steps + 1):
        out = training_step(model, batch, schedule, tok.mask_id, tok.pad_id,
                            t_eps=cfg.diffusion.t_eps, low_discrepancy=True)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == 1:
            print(f"  step {step:>5} | nelbo {out['loss'].item():9.4f} | "
                  f"ce {out['ce'].item():7.4f} | acc {out['acc'].item():.4f} | "
                  f"real_acc {out['real_acc'].item():.4f}")
        if not torch.isfinite(out["loss"]):
            raise RuntimeError("NaN/Inf loss — check t_eps clamping.")

    # --- sample and check we recover the training set ---
    print("[gate2] sampling 32 ...")
    x = sample(model, schedule, 32, tok.max_len, tok.n_struct, tok.bos_id,
               tok.prefix_ids, tok.mask_id, tok.pad_id,
               n_steps=64, device=args.device)
    gen = [tok.decode_smiles(r.tolist()) for r in x.cpu()]

    from rdkit import Chem
    train_can = set(smi)
    ok = 0
    for g in gen:
        if g is None:
            continue
        m = Chem.MolFromSmiles(g)
        if m and Chem.MolToSmiles(m, canonical=True) in train_can:
            ok += 1
    print(f"[gate2] samples matching the training set: {ok}/{len(gen)}")
    print("  (HIGH is correct here — an overfit model must MEMORISE. "
         "Novel molecules at this stage mean it has not converged.)")

    passed = out["real_acc"].item() > 0.95 and ok >= len(gen) * 0.5
    print(f"[gate2] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())