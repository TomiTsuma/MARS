"""
Hook layer: the Phase 2 entry points, built in Phase 1.

Every transformer block exposes exactly two hook sites on the residual stream:

    READ  -> ActivationCache.record(layer, x)   : extraction (Phase 2 §6.2)
    WRITE -> SteerFn(x, layer, step)            : intervention (Phase 2 §6.4)

Both default to None, so Phase 1 runs completely unaffected. Before trusting a
single Phase 2 number, run `verify_noop_identity` and confirm that enabling the
hooks with no-ops changes nothing bit-for-bit. If it does not hold, every
steering result is confounded by the instrumentation itself.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Protocol

import torch


class SteerFn(Protocol):
    """Signature of a Phase 2 intervention.

    Called INSIDE each block, immediately before the block's computation.

        x     : (B, L, D) residual stream
        layer : int, 0-indexed
        step  : int or None, reverse-diffusion step index (None during training)

    Must return a tensor of the same shape. Phase 2's additive primitive is

        return x + alpha * v[layer]

    with alpha expressed in units of that layer's residual-stream std — see
    ResidualStats below. A fixed absolute alpha is not comparable across depth.
    """
    def __call__(self, x: torch.Tensor, layer: int,
                 step: Optional[int]) -> torch.Tensor: ...


class ActivationCache:
    """Records residual-stream activations keyed by (layer, step).

    Logging policy is deliberately asymmetric, because exhaustive capture is
    infeasible (10k molecules x N steps x L layers):

        extraction  -> full tensors, small contrastive sets, `mode='full'`
        application -> summary statistics only,               `mode='stats'`
    """

    def __init__(self, mode: str = "full", layers: Optional[List[int]] = None,
                 positions: Optional[List[int]] = None, device: str = "cpu"):
        assert mode in ("full", "stats", "off")
        self.mode = mode
        self.layers = set(layers) if layers is not None else None
        self.positions = positions
        self.device = device
        self.step: Optional[int] = None
        self.store: Dict[tuple, List[torch.Tensor]] = defaultdict(list)

    def set_step(self, step: Optional[int]) -> None:
        self.step = step

    def record(self, layer: int, x: torch.Tensor) -> None:
        if self.mode == "off":
            return
        if self.layers is not None and layer not in self.layers:
            return
        key = (layer, self.step)
        h = x.detach()
        if self.positions is not None:
            h = h[:, self.positions, :]
        if self.mode == "full":
            self.store[key].append(h.to(self.device))
        else:  # stats
            self.store[key].append(torch.stack([
                h.mean(), h.std(), h.norm(dim=-1).mean()
            ]).to(self.device))

    def stack(self, layer: int, step: Optional[int] = None) -> torch.Tensor:
        return torch.cat(self.store[(layer, step)], dim=0)

    def clear(self) -> None:
        self.store.clear()


class ResidualStats:
    """Per-layer residual-stream standard deviation.

    Required for Phase 2: residual norms grow with depth in a pre-LN
    transformer, so a fixed absolute steering coefficient is a different
    effective dose at every layer, and the depth-localisation result (E4)
    becomes uninterpretable. Fit this once on the validation set and express
    all coefficients in these units.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.std = torch.ones(n_layers)

    @torch.no_grad()
    def fit(self, model, loader, mask_id: int, schedule, device: str,
            n_batches: int = 20) -> "ResidualStats":
        from objective import forward_mask
        from schedule import sample_t

        cache = ActivationCache(mode="full", device="cpu")
        model.eval()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            ids = batch["ids"].to(device)
            maskable = batch["maskable"].to(device)
            t = sample_t(ids.shape[0], device)
            x_t, _ = forward_mask(ids, maskable, t, schedule, mask_id)
            model(x_t, t, cache=cache)
        for l in range(self.n_layers):
            self.std[l] = cache.stack(l).float().std()
        cache.clear()
        return self


@torch.no_grad()
def verify_noop_identity(model, x: torch.Tensor, t: torch.Tensor,
                         atol: float = 0.0) -> bool:
    """Assert that instrumentation is inert.

    Runs the model three ways — no hooks, cache only, cache + identity steer —
    and requires bit-identical logits. Run this once after implementing hooks
    and again after any change to the block. It is the cheapest possible
    guarantee that Phase 2 measures the model and not the measuring apparatus.
    """
    model.eval()
    base = model(x, t)

    cache = ActivationCache(mode="full", device="cpu")
    with_cache = model(x, t, cache=cache)

    identity: SteerFn = lambda h, layer, step: h
    with_both = model(x, t, steer=identity, cache=cache)

    ok = torch.equal(base, with_cache) and torch.equal(base, with_both)
    if not ok:
        d1 = (base - with_cache).abs().max().item()
        d2 = (base - with_both).abs().max().item()
        print(f"NO-OP IDENTITY FAILED  max|d_cache|={d1:.3e}  max|d_steer|={d2:.3e}")
    else:
        print("no-op identity OK (hooks are inert)")
    return ok
