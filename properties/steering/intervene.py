"""
Intervention primitives — the SteerFn implementations.

THE DEFAULT IS ADDITION, NOT PROJECTION
---------------------------------------
The precedent literature (Arditi et al. 2024; Shnaidman et al. 2026) applies
directions by orthogonal projection, because refusal is a behaviour to be
REMOVED. This work injects a graded target property, so the correct primitive
is addition with a signed, scaled coefficient. Inheriting the projection
formula by default silently wastes an entire experiment.

COEFFICIENT NORMALISATION IS MANDATORY
--------------------------------------
Residual-stream norms grow with depth in a pre-layer-norm transformer. A fixed
absolute alpha is therefore a different effective dose at every layer, and any
cross-layer comparison under a fixed alpha is confounded. All coefficients here
are expressed in units of the target layer's empirical residual-stream standard
deviation. Without this the depth-localisation results are not interpretable.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from properties.steering.positions import AllPositions, PositionSet
from properties.steering.schedules import AllSteps, Schedule


class _BaseSteer:
    """Common context handling for every intervention.

    The sampler calls set_context(x, step, n_steps) before each forward pass so
    that position controllers needing token identity (MaskedOnly, FrozenOnly)
    can build their masks. During training or a plain forward pass no context
    is supplied and those controllers raise rather than silently misbehave.
    """

    def __init__(self, layers, schedule, positions, stats, n_struct,
                 pad_id, mask_id):
        self.layers = set(layers) if layers is not None else None   # None = all
        self.schedule: Schedule = schedule or AllSteps()
        self.positions: PositionSet = positions or AllPositions()
        self.stats = stats
        self.n_struct = n_struct
        self.pad_id = pad_id
        self.mask_id = mask_id
        self._ids: Optional[torch.Tensor] = None
        self._step: Optional[int] = None
        self._n_steps: Optional[int] = None
        self.n_calls = 0          # debug counter: should equal n_layers*n_steps
        self.n_applied = 0        # hook invocations that passed the gate
        self.mass = 0             # total (batch x position) elements modified —
                                  # the denominator for intervention-mass
                                  # normalisation in the E3 localisation control

    def set_context(self, ids: torch.Tensor, step: Optional[int] = None,
                    n_steps: Optional[int] = None) -> None:
        self._ids, self._step, self._n_steps = ids, step, n_steps

    def _sigma(self, layer: int) -> float:
        if self.stats is None:
            return 1.0
        std = getattr(self.stats, "std", self.stats)
        return float(std[layer])

    def _gate(self, layer: int, step: Optional[int]) -> bool:
        self.n_calls += 1
        if self.layers is not None and layer not in self.layers:
            return False
        step = self._step if step is None else step
        return self.schedule.active(step, self._n_steps)

    def _mask(self, h: torch.Tensor) -> torch.Tensor:
        m = self.positions.mask(self._ids, self.n_struct, self.pad_id,
                                self.mask_id, h.shape)
        self.mass += int(m.sum().item())
        return m.to(h.dtype)

    def describe(self) -> Dict:
        return {"layers": sorted(self.layers) if self.layers else "all",
                "schedule": self.schedule.name,
                "positions": self.positions.name,
                "n_calls": self.n_calls, "n_applied": self.n_applied,
                "mass": self.mass}

    def reset_counters(self):
        self.n_calls = self.n_applied = self.mass = 0


class AdditiveSteer(_BaseSteer):
    """h <- h + alpha * sigma_layer * v      (PRIMARY)

    Signed alpha gives bidirectional control: positive pushes the property up,
    negative pushes it down, zero recovers the unconditional model. If negative
    alpha does not reduce the property, that is evidence about linearity, not a
    bug to be tuned away.
    """

    def __init__(self, direction, alpha_sigma: float, stats=None, layers=None,
                 schedule=None, positions=None, n_struct: int = 9,
                 pad_id: int = 0, mask_id: int = 1):
        super().__init__(layers, schedule, positions, stats, n_struct, pad_id, mask_id)
        v = torch.as_tensor(direction, dtype=torch.float32)
        self.v = v / v.norm().clamp(min=1e-8)
        self.alpha_sigma = float(alpha_sigma)

    def __call__(self, h: torch.Tensor, layer: int, step: Optional[int] = None):
        if not self._gate(layer, step) or self.alpha_sigma == 0.0:
            return h
        self.n_applied += 1
        scale = self.alpha_sigma * self._sigma(layer)
        v = self.v.to(device=h.device, dtype=h.dtype)
        return h + self._mask(h) * scale * v


class ProjectiveSteer(_BaseSteer):
    """h <- h - <h, v> v      (ABLATION ARM ONLY)

    Removes the component of the representation along the direction. Answers
    the complementary question — does ablating this direction degrade the
    model's ability to express the property? — and provides comparability with
    the text precedent. Not for injecting a target value.
    """

    def __init__(self, direction, stats=None, layers=None, schedule=None,
                 positions=None, n_struct: int = 9, pad_id: int = 0,
                 mask_id: int = 1, strength: float = 1.0):
        super().__init__(layers, schedule, positions, stats, n_struct, pad_id, mask_id)
        v = torch.as_tensor(direction, dtype=torch.float32)
        self.v = v / v.norm().clamp(min=1e-8)
        self.strength = float(strength)

    def __call__(self, h, layer, step=None):
        if not self._gate(layer, step):
            return h
        self.n_applied += 1
        v = self.v.to(device=h.device, dtype=h.dtype)
        proj = (h * v).sum(-1, keepdim=True) * v
        return h - self._mask(h) * self.strength * proj


class ComposedSteer(_BaseSteer):
    """h <- h + sum_i alpha_i * sigma_layer * v_i      (E7)

    Multi-objective conditioning by linear combination. Whether this yields
    partially independent control or interference is the experimental question;
    the correlation structure between the chosen properties determines what to
    expect, so the E7 grid spans near-orthogonal, correlated and adversarial
    pairs deliberately.
    """

    def __init__(self, directions: Sequence, alphas: Sequence[float], stats=None,
                 layers=None, schedule=None, positions=None, n_struct: int = 9,
                 pad_id: int = 0, mask_id: int = 1):
        super().__init__(layers, schedule, positions, stats, n_struct, pad_id, mask_id)
        vs = [torch.as_tensor(d, dtype=torch.float32) for d in directions]
        self.vs = [v / v.norm().clamp(min=1e-8) for v in vs]
        self.alphas = [float(a) for a in alphas]
        assert len(self.vs) == len(self.alphas)

    def __call__(self, h, layer, step=None):
        if not self._gate(layer, step):
            return h
        self.n_applied += 1
        sig = self._sigma(layer)
        m = self._mask(h)
        delta = torch.zeros_like(h)
        for v, a in zip(self.vs, self.alphas):
            if a == 0.0:
                continue
            delta = delta + a * sig * v.to(device=h.device, dtype=h.dtype)
        return h + m * delta


class NullSteer(_BaseSteer):
    """Identity intervention. Used to verify that the machinery itself is inert
    — generation with NullSteer must match generation with steer=None."""

    def __init__(self, **kw):
        super().__init__(None, None, None, None, kw.get("n_struct", 9),
                         kw.get("pad_id", 0), kw.get("mask_id", 1))

    def __call__(self, h, layer, step=None):
        self.n_calls += 1
        return h


def build_steer(kind: str, direction=None, alpha: float = 0.0, **kw):
    kind = (kind or "additive").lower()
    if kind in ("none", "null"):
        return NullSteer(**kw)
    if kind == "additive":
        return AdditiveSteer(direction, alpha, **kw)
    if kind in ("projective", "ablate"):
        return ProjectiveSteer(direction, **kw)
    raise ValueError(f"unknown steer kind {kind!r}")