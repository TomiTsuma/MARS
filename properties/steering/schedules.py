"""
The WHEN axis — which reverse-diffusion steps receive the intervention.

This axis exists only because generation is iterative: the denoiser is
re-invoked at every one of the N reverse steps, so an intervention can be
scheduled in time. There is no autoregressive analogue.

A confound to keep in view throughout: because tokens freeze as they unmask,
the set of positions an intervention can causally affect shrinks monotonically
over the trajectory. Early-step dominance may therefore be mechanical rather
than representational. Separating the two requires holding the unmasking
schedule fixed while varying the steering schedule — see `p2_localize.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class Schedule(Protocol):
    name: str
    def active(self, step: Optional[int], n_steps: Optional[int] = None) -> bool: ...


@dataclass
class AllSteps:
    """Intervene at every reverse step. The default and the reference."""
    name: str = "all"
    def active(self, step=None, n_steps=None) -> bool:
        return True


@dataclass
class FirstK:
    """Intervene only during the first k denoising steps.

    The precedent (Shnaidman et al. 2026) reports that this dominates for
    refusal steering in text; the 3D molecular literature reports an analogous
    exploration phase early in the trajectory. Both are priors to test, not
    facts to assume: in a 1D chemical language model 'early' means composition
    and scaffold, not geometry.
    """
    k: int = 8
    name: str = "first_k"
    def active(self, step=None, n_steps=None) -> bool:
        return step is not None and step < self.k


@dataclass
class LastK:
    """Intervene only during the final k steps. Expected to be weak; a flat
    LAST-k curve alongside a steep FIRST-k curve is the signature of early-step
    dominance."""
    k: int = 8
    name: str = "last_k"
    def active(self, step=None, n_steps=None) -> bool:
        if step is None or n_steps is None:
            return False
        return step >= n_steps - self.k


@dataclass
class EveryK:
    """Periodic intervention — an intermediate condition that separates
    'how many steps' from 'which steps'."""
    k: int = 4
    name: str = "every_k"
    def active(self, step=None, n_steps=None) -> bool:
        return step is not None and step % self.k == 0


@dataclass
class StepWindow:
    """Arbitrary contiguous window [lo, hi). Used by the property-timing
    correspondence experiment, where each property's critical window is
    located rather than assumed."""
    lo: int = 0
    hi: int = 8
    name: str = "window"
    def active(self, step=None, n_steps=None) -> bool:
        return step is not None and self.lo <= step < self.hi


@dataclass
class NoSteps:
    """Null controller — the unconditional reference path."""
    name: str = "none"
    def active(self, step=None, n_steps=None) -> bool:
        return False


def from_spec(spec: str) -> Schedule:
    """Parse a CLI schedule spec.

        all | none | first:8 | last:8 | every:4 | window:0:16
    """
    spec = (spec or "all").strip().lower()
    if spec in ("all", ""):
        return AllSteps()
    if spec == "none":
        return NoSteps()
    head, *rest = spec.split(":")
    if head == "first":
        return FirstK(int(rest[0]))
    if head == "last":
        return LastK(int(rest[0]))
    if head == "every":
        return EveryK(int(rest[0]))
    if head == "window":
        return StepWindow(int(rest[0]), int(rest[1]))
    raise ValueError(f"unknown schedule spec: {spec!r}")


def sweep_specs(n_steps: int, kappas=(1, 2, 4, 8, 16, 32)) -> list:
    """The standard E3 schedule ablation grid."""
    out = ["all"]
    for k in kappas:
        if k <= n_steps:
            out += [f"first:{k}", f"last:{k}", f"every:{max(1, n_steps // k)}"]
    return out