"""
Central configuration objects for Phase 1.

Everything that defines a checkpoint's identity lives here. Two rules:

  1. Fields marked IRREVERSIBLE are baked into the trained weights. Changing
     them invalidates every checkpoint trained under the old value.
  2. Every experiment is defined by a serialised config, never by ad-hoc
     command-line flags. `Config.to_json` / `from_json` exist for this reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime


@dataclass
class DataConfig:
    corpus: str = "zinc250k"            # zinc250k | moses | guacamol | chembl
    max_len: int = 128                  # IRREVERSIBLE (positional capacity)
    n_prefix: int = 8                   # IRREVERSIBLE (structural prefix width)
    alphabet_path: str = "artifacts/alphabet.json"
    processed_dir: str = "artifacts/processed"
    scaffold_split: bool = True
    val_frac: float = 0.05
    test_frac: float = 0.05
    seed: int = 0


@dataclass
class ModelConfig:
    # Config-S defaults; Config-B is d_model=768, n_layers=14, n_heads=12.
    d_model: int = 640                  # IRREVERSIBLE
    n_layers: int = 12                  # IRREVERSIBLE
    n_heads: int = 10                   # IRREVERSIBLE
    d_ff: int = 2560                    # IRREVERSIBLE (SwiGLU inner dim)
    dropout: float = 0.05
    time_embed_dim: int = 256
    rope_base: float = 10_000.0
    use_adaln: bool = True              # IRREVERSIBLE-ish: this is the Phase 2
                                        # learned-conditioning baseline port.
                                        # Do not remove as "dead weight".

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        return self.d_model // self.n_heads


@dataclass
class DiffusionConfig:
    schedule: str = "loglinear"         # loglinear | cosine
    t_eps: float = 1e-3                 # clamp on t; guards the 1/t loss weight
    low_discrepancy: bool = True        # stratified t sampling across the batch
    n_sampling_steps: int = 128         # inference-time only, freely tunable
    unmasking: str = "ancestral"        # ancestral | confidence
                                        # Keep 'ancestral' canonical: confidence
                                        # unmasking makes the order data-dependent
                                        # and entangles Phase 2 WHEN-axis results.


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 2_500
    max_steps: int = 200_000
    ema_decay: float = 0.9999           # evaluate the EMA copy, always
    log_every: int = 100
    eval_every: int = 5_000
    ckpt_every: int = 10_000
    gate_every: int = 1000            # steps between Tier 0 gate checks
    gate_n_samples: int = 20         # molecules sampled per gate check
    precision: str = "bf16"
    out_dir: str = f"runs/phase1/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seed: int = 0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return Config(
            data=DataConfig(**d["data"]),
            model=ModelConfig(**d["model"]),
            diffusion=DiffusionConfig(**d["diffusion"]),
            train=TrainConfig(**d["train"]),
        )


def config_b() -> Config:
    """Config-B (~90M). Everything else identical to Config-S."""
    c = Config()
    c.model.d_model = 768
    c.model.n_layers = 14
    c.model.n_heads = 12
    c.model.d_ff = 3072
    return c
