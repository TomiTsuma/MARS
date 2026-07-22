"""
Training entry point: unconditional MDLM pretraining on ZINC250k (SELFIES).

Expects `scripts/prepare.py` to have already produced, under
`--processed-dir` (default `artifacts/processed`):

    tokenizer.json, train.npy, val.npy

Usage:
    python scripts/train.py
    python scripts/train.py --config-b --out-dir runs/phase1_configb
    python scripts/train.py --resume runs/phase1/latest.pt

Every run writes, under `--out-dir`:

    config.json        the fully-resolved Config, for exact reproduction
    train_log.jsonl     one JSON record per logged/evaluated step
    ckpt_{step}.pt       periodic checkpoints (model, EMA, optimizer, step)
    latest.pt            copy of the most recent checkpoint, for --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config, config_b
from datasets.dataset import make_loader
from datasets.tokenizer import SelfiesTokenizer
from model.model import MDLM
from model.objective import training_step
from model.schedule import get_schedule


# ------------------------------------------------------------------- EMA
class EMA:
    """Shadow copy of model weights, updated by exponential moving average.

    config/config.py is explicit that the EMA copy, not the raw weights, is
    what gets evaluated and sampled from — see TrainConfig.ema_decay.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}


# ------------------------------------------------------------------- LR schedule
def lr_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup, then cosine decay to 0 at max_steps."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                     betas: tuple) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matrices (>=2D), not to norms,
    biases, or embeddings-as-vectors. Standard practice; prevents decaying
    away the RMSNorm gains and adaLN-zero biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# ------------------------------------------------------------------- data
def load_data(data_cfg, train_cfg, num_workers: int):
    tok = SelfiesTokenizer.load(os.path.join(data_cfg.processed_dir, "tokenizer.json"))
    train_ids = np.load(os.path.join(data_cfg.processed_dir, "train.npy"))
    val_ids = np.load(os.path.join(data_cfg.processed_dir, "val.npy"))

    train_loader = make_loader(train_ids, tok.pad_id, tok.n_struct,
                               train_cfg.batch_size, shuffle=True,
                               num_workers=num_workers, seed=train_cfg.seed)
    val_loader = make_loader(val_ids, tok.pad_id, tok.n_struct,
                             train_cfg.batch_size, shuffle=False,
                             num_workers=num_workers, seed=train_cfg.seed)
    return tok, train_loader, val_loader


# ------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, schedule, mask_id, pad_id, t_eps, low_discrepancy,
            device, max_batches: int = 50) -> dict:
    model.eval()
    totals = {"loss": 0.0, "ce": 0.0, "acc": 0.0, "real_acc": 0.0, "mask_frac": 0.0}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = training_step(model, batch, schedule, mask_id, pad_id, t_eps, low_discrepancy)
        for k in totals:
            totals[k] += out[k].item()
        n += 1
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def evaluate_ema(model, ema: EMA, loader, schedule, mask_id, pad_id, t_eps,
                 low_discrepancy, device, max_batches: int = 50) -> dict:
    """Swap the EMA shadow weights in, evaluate, swap the live weights back."""
    backup = deepcopy(model.state_dict())
    model.load_state_dict(ema.state_dict())
    metrics = evaluate(model, loader, schedule, mask_id, pad_id, t_eps,
                       low_discrepancy, device, max_batches)
    model.load_state_dict(backup)
    return metrics


# ------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, step: int, model, ema: EMA, optimizer,
                    scheduler, config: Config) -> None:
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": json.dumps({
            "data": config.data.__dict__, "model": config.model.__dict__,
            "diffusion": config.diffusion.__dict__, "train": config.train.__dict__,
        }),
    }, path)


def load_checkpoint(path: str, model, ema: EMA, optimizer, scheduler,
                    device) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]


# ------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON Config (Config.to_json output). "
                             "Overrides Config-S/-B defaults; CLI flags below "
                             "override this in turn.")
    parser.add_argument("--config-b", action="store_true",
                        help="Use Config-B (~90M) instead of Config-S.")
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ckpt-every", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None,
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint (e.g. runs/phase1/latest.pt) to resume from.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model. Off by default: the Triton "
                             "backend it needs is unreliable on Windows.")
    args = parser.parse_args()

    # ---- resolve config: base -> --config JSON -> individual CLI overrides
    #
    # On --resume without an explicit --config, load the config.json saved
    # next to the checkpoint rather than falling back to Config() defaults.
    # ModelConfig fields are IRREVERSIBLE (baked into the weights); silently
    # resuming under a different config than the one that produced the
    # checkpoint corrupts the run without any visible error.
    if args.config:
        config_path = args.config
    elif args.resume:
        config_path = os.path.join(os.path.dirname(args.resume), "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"--resume given but no config.json found next to it at {config_path}. "
                "Pass --config explicitly to specify the architecture to resume into.")
        print(f"--resume without --config: loading {config_path}")
    else:
        config_path = None
    config = Config.from_json(config_path) if config_path else (config_b() if args.config_b else Config())
    if args.processed_dir:
        config.data.processed_dir = args.processed_dir
    if args.out_dir:
        config.train.out_dir = args.out_dir
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr
    if args.max_steps:
        config.train.max_steps = args.max_steps
    if args.warmup_steps:
        config.train.warmup_steps = args.warmup_steps
    if args.log_every:
        config.train.log_every = args.log_every
    if args.eval_every:
        config.train.eval_every = args.eval_every
    if args.ckpt_every:
        config.train.ckpt_every = args.ckpt_every
    if args.precision:
        config.train.precision = args.precision
    if args.seed is not None:
        config.train.seed = args.seed
        config.data.seed = args.seed

    device = torch.device(args.device)
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)

    os.makedirs(config.train.out_dir, exist_ok=True)
    config.to_json(os.path.join(config.train.out_dir, "config.json"))

    # ---- data
    tok, train_loader, val_loader = load_data(config.data, config.train, args.num_workers)
    print(f"vocab_size={tok.vocab_size}  n_struct={tok.n_struct}  "
         f"train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}")

    # ---- model
    model = MDLM(vocab_size=tok.vocab_size, cfg=config.model, pad_id=tok.pad_id).to(device)
    print("param breakdown:", model.param_breakdown())
    if args.compile:
        model = torch.compile(model)

    schedule = get_schedule(config.diffusion.schedule)
    ema = EMA(model, config.train.ema_decay)
    optimizer = build_optimizer(model, config.train.lr, config.train.weight_decay,
                                config.train.betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, config.train.warmup_steps, config.train.max_steps))

    precision = config.train.precision
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))

    step = 0
    if args.resume:
        step = load_checkpoint(args.resume, model, ema, optimizer, scheduler, device)
        print(f"resumed from {args.resume} at step {step}")

    log_path = os.path.join(config.train.out_dir, "train_log.jsonl")
    log_file = open(log_path, "a")

    def log_record(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    # ---- training loop
    data_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    while step < config.train.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = training_step(model, batch, schedule, tok.mask_id, tok.pad_id,
                                config.diffusion.t_eps, config.diffusion.low_discrepancy)
            loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
        scheduler.step()
        ema.update(model)
        step += 1

        if step % config.train.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = config.train.log_every / max(elapsed, 1e-9)
            record = {
                "step": step, "split": "train",
                "loss": out["loss"].item(), "ce": out["ce"].item(),
                "acc": out["acc"].item(), "real_acc": out["real_acc"].item(),
                "mask_frac": out["mask_frac"].item(),
                "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                "steps_per_sec": steps_per_sec,
            }
            log_record(record)
            print(f"step {step:>7,}  loss {record['loss']:.4f}  ce {record['ce']:.4f}  "
                 f"real_acc {record['real_acc']:.4f}  lr {record['lr']:.2e}  "
                 f"grad_norm {record['grad_norm']:.2f}  {steps_per_sec:.2f} it/s")
            t0 = time.time()

        if step % config.train.eval_every == 0 or step == config.train.max_steps:
            val_metrics = evaluate(model, val_loader, schedule, tok.mask_id, tok.pad_id,
                                   config.diffusion.t_eps, config.diffusion.low_discrepancy, device)
            ema_metrics = evaluate_ema(model, ema, val_loader, schedule, tok.mask_id, tok.pad_id,
                                       config.diffusion.t_eps, config.diffusion.low_discrepancy, device)
            log_record({"step": step, "split": "val", **val_metrics})
            log_record({"step": step, "split": "val_ema", **ema_metrics})
            print(f"  [val]     step {step:>7,}  loss {val_metrics['loss']:.4f}  "
                 f"real_acc {val_metrics['real_acc']:.4f}")
            print(f"  [val_ema] step {step:>7,}  loss {ema_metrics['loss']:.4f}  "
                 f"real_acc {ema_metrics['real_acc']:.4f}")
            model.train()

        if step % config.train.ckpt_every == 0 or step == config.train.max_steps:
            ckpt_path = os.path.join(config.train.out_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(ckpt_path, step, model, ema, optimizer, scheduler, config)
            save_checkpoint(os.path.join(config.train.out_dir, "latest.pt"),
                            step, model, ema, optimizer, scheduler, config)
            print(f"  saved checkpoint: {ckpt_path}")

    log_file.close()
    print(f"done: {config.train.max_steps:,} steps -> {config.train.out_dir}")


if __name__ == "__main__":
    main()
