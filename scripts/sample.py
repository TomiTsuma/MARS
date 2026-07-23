"""
Sampling entry point: draw unconditional molecules from a trained MDLM
checkpoint and (optionally) score them against the Tier 0 gate.

Loads the EMA weights by default — config/config.py and model/metrics.py are
both explicit that the EMA copy, not the raw weights, is the one to trust.

Usage:
    python scripts/sample.py --checkpoint runs/phase1/latest.pt --n-samples 1000
    python scripts/sample.py --checkpoint runs/phase1/latest.pt --n-samples 10000 --eval
"""
from __future__ import annotations

import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from datasets.tokenizer import SelfiesTokenizer
from model.model import MDLM
from model.sampler import sample_to_smiles
from model.schedule import get_schedule
import model.metrics as metrics


def load_model(checkpoint_path: str, config: Config, tok: SelfiesTokenizer,
              device: torch.device, use_raw: bool) -> MDLM:
    model = MDLM(vocab_size=tok.vocab_size, cfg=config.model, pad_id=tok.pad_id).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"] if use_raw else ckpt["ema"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a checkpoint written by scripts/train.py, "
                             "e.g. runs/phase1/latest.pt")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json. Defaults to config.json next "
                             "to the checkpoint (same convention as train.py --resume).")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Override config.data.processed_dir (tokenizer + splits).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file for generated SMILES, one per line. "
                             "Defaults to <checkpoint_dir>/samples.smi")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Reverse-diffusion steps. Defaults to config.diffusion.n_sampling_steps.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default=None, choices=["ancestral", "confidence"],
                        help="Defaults to config.diffusion.unmasking.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-raw", action="store_true",
                        help="Sample from the raw (non-EMA) weights instead.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval", action="store_true",
                        help="Score the generated set against train/test SMILES "
                             "(validity, uniqueness, novelty, FCD, Tier 0 gate).")
    args = parser.parse_args()

    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found at {config_path}. Pass --config explicitly.")
    config = Config.from_json(config_path)
    if args.processed_dir:
        config.data.processed_dir = args.processed_dir

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    tok = SelfiesTokenizer.load(os.path.join(config.data.processed_dir, "tokenizer.json"))
    model = load_model(args.checkpoint, config, tok, device, args.use_raw)
    schedule = get_schedule(config.diffusion.schedule)

    n_steps = args.n_steps or config.diffusion.n_sampling_steps
    mode = args.mode or config.diffusion.unmasking
    print(f"sampling {args.n_samples:,} molecules  "
         f"({'raw' if args.use_raw else 'EMA'} weights, {n_steps} steps, mode={mode})")

    generated = sample_to_smiles(
        model, schedule, tok, args.n_samples, n_steps=n_steps, device=str(device),
        batch_size=args.batch_size, temperature=args.temperature, mode=mode)

    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint), "samples.smi")
    with open(out_path, "w") as f:
        f.write("\n".join(s for s in generated if s is not None))
    n_valid = sum(1 for s in generated if s is not None)
    print(f"wrote {n_valid:,}/{len(generated):,} decodable SMILES -> {out_path}")

    if args.eval:
        with open(os.path.join(config.data.processed_dir, "train_smiles.txt")) as f:
            train_smiles = f.read().splitlines()
        with open(os.path.join(config.data.processed_dir, "test_smiles.txt")) as f:
            test_smiles = f.read().splitlines()

        report = metrics.evaluate(generated, train_smiles, test_smiles)
        print("\nmetrics:")
        for k, v in report.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        metrics.tier0_gate(report)


if __name__ == "__main__":
    main()
