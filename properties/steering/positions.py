"""
The WHERE-position axis — which token positions receive the intervention.

Four conditions, one of which has no counterpart in autoregressive steering.

MaskedOnly is the diffusion-native case and the most interesting. Because a
token is frozen once unmasked, steering an already-revealed position cannot
change that token at all — its only effect is indirect, propagating through
attention to positions that remain masked. FrozenOnly and MaskedOnly therefore
probe two genuinely distinct causal pathways, and the interaction between the
steering schedule and the unmasking schedule is, as far as the literature goes,
unexamined.

All controllers return a (B, L, 1) boolean mask so the intervention can be
applied by broadcasting against the (B, L, D) residual stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


class PositionSet(Protocol):
    name: str
    def mask(self, ids: Optional[torch.Tensor], n_struct: int,
             pad_id: int, mask_id: int, shape) -> torch.Tensor: ...


def _ones(shape, device, dtype=torch.bool):
    b, l = shape[0], shape[1]
    return torch.ones(b, l, 1, dtype=dtype, device=device)


@dataclass
class AllPositions:
    """Everything except padding. The default."""
    name: str = "all"
    def mask(self, ids, n_struct, pad_id, mask_id, shape):
        if ids is None:
            return _ones(shape, shape[-1] if torch.is_tensor(shape) else "cpu")
        return ids.ne(pad_id).unsqueeze(-1)


@dataclass
class PrefixOnly:
    """The structural prefix (BOS + P0..P7) only.

    Note the asymmetry with extraction: a direction extracted at the prefix can
    be applied anywhere, and applying it only at the prefix tests whether those
    positions are a sufficient causal route as well as a readable one.
    """
    name: str = "prefix"
    def mask(self, ids, n_struct, pad_id, mask_id, shape):
        b, l = shape[0], shape[1]
        dev = ids.device if ids is not None else "cpu"
        pos = torch.arange(l, device=dev)
        return (pos < n_struct).view(1, l, 1).expand(b, l, 1)


@dataclass
class BodyOnly:
    """Molecule tokens only — excludes prefix and padding."""
    name: str = "body"
    def mask(self, ids, n_struct, pad_id, mask_id, shape):
        b, l = shape[0], shape[1]
        dev = ids.device if ids is not None else "cpu"
        pos = torch.arange(l, device=dev)
        m = (pos >= n_struct).view(1, l, 1).expand(b, l, 1)
        if ids is not None:
            m = m & ids.ne(pad_id).unsqueeze(-1)
        return m


@dataclass
class MaskedOnly:
    """Positions currently holding [MASK] — DIFFUSION-NATIVE.

    These are the only positions whose token identity the intervention can
    still change. Requires the sampler to supply context via set_context().
    """
    name: str = "masked"
    def mask(self, ids, n_struct, pad_id, mask_id, shape):
        if ids is None:
            raise RuntimeError(
                "MaskedOnly requires token context. The sampler must call "
                "steer.set_context(x, step, n_steps) before the model forward "
                "pass — see diffusion/sampler.py.")
        return ids.eq(mask_id).unsqueeze(-1)


@dataclass
class FrozenOnly:
    """Already-revealed body positions. Steering these CANNOT change their own
    tokens; any effect is purely indirect, through attention on still-masked
    neighbours. The complement of MaskedOnly, and the cleanest way to isolate
    the indirect pathway."""
    name: str = "frozen"
    def mask(self, ids, n_struct, pad_id, mask_id, shape):
        if ids is None:
            raise RuntimeError("FrozenOnly requires token context (set_context).")
        b, l = ids.shape
        pos = torch.arange(l, device=ids.device)
        body = (pos >= n_struct).view(1, l).expand(b, l)
        revealed = ids.ne(mask_id) & ids.ne(pad_id) & body
        return revealed.unsqueeze(-1)


REGISTRY = {c.name: c for c in
            (AllPositions(), PrefixOnly(), BodyOnly(), MaskedOnly(), FrozenOnly())}


def from_spec(spec: str) -> PositionSet:
    spec = (spec or "all").strip().lower()
    if spec not in REGISTRY:
        raise ValueError(f"unknown position spec {spec!r}; "
                         f"known: {sorted(REGISTRY)}")
    return REGISTRY[spec]


SWEEP_SPECS = ("all", "prefix", "masked", "frozen")