"""
Noise schedules for absorbing-state discrete diffusion.

Convention (follow it exactly, sign errors here are painful to find):

    alpha_t = P(a token is still UNMASKED at time t)
    alpha_0 = 1  (clean)      alpha_1 = 0  (fully masked)

The MDLM continuous-time NELBO carries the weight

    w(t) = -alpha'_t / (1 - alpha_t)

Log-linear schedule (the MDLM default; "log-linear noise" sigma_t = -log(1-t)
is exactly alpha_t = 1 - t):

    alpha_t = 1 - t     ->   w(t) = 1 / t

That 1/t is the entire reason `t_eps` and low-discrepancy sampling exist: the
weight diverges as t -> 0 and will produce loss spikes or NaNs if unclamped.

Cosine schedule (control arm for E4, so temporal-asymmetry findings can be
shown not to be an artefact of schedule shape):

    alpha_t = cos^2(pi t / 2)   ->   w(t) = pi * cot(pi t / 2)
"""
from __future__ import annotations

import math
import torch


class NoiseSchedule:
    """Base class. Subclasses implement alpha and weight."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def weight(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Probability a given token is masked at time t."""
        return 1.0 - self.alpha(t)

    def posterior_unmask_prob(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """For the reverse step t -> s (s < t): probability that a currently
        masked token is revealed at this step.

            p = (alpha_s - alpha_t) / (1 - alpha_t)
        """
        a_t, a_s = self.alpha(t), self.alpha(s)
        return ((a_s - a_t) / (1.0 - a_t).clamp(min=1e-8)).clamp(0.0, 1.0)


class LogLinearSchedule(NoiseSchedule):
    def alpha(self, t):
        return 1.0 - t

    def weight(self, t):
        return 1.0 / t


class CosineSchedule(NoiseSchedule):
    def alpha(self, t):
        return torch.cos(math.pi * t / 2.0) ** 2

    def weight(self, t):
        half = math.pi * t / 2.0
        return math.pi * torch.cos(half) / torch.sin(half).clamp(min=1e-8)


def get_schedule(name: str) -> NoiseSchedule:
    return {"loglinear": LogLinearSchedule, "cosine": CosineSchedule}[name]()


# ------------------------------------------------------------------ t sampling
def sample_t(batch_size: int, device, t_eps: float = 1e-3,
             low_discrepancy: bool = True) -> torch.Tensor:
    """Sample diffusion times for a batch.

    Low-discrepancy (stratified) sampling places one t in each of B equal bins,
    which materially reduces gradient variance versus i.i.d. uniform. MDLM does
    this; it is not a nicety, it visibly stabilises early training.
    """
    if low_discrepancy:
        u = torch.rand(1, device=device)
        idx = torch.arange(batch_size, device=device)
        t = (idx + u) / batch_size
    else:
        t = torch.rand(batch_size, device=device)
    return t.clamp(min=t_eps, max=1.0)

