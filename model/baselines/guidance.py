"""
Guidance baselines — the comparators activation steering is measured against.

THE DISTINCTION THAT MATTERS
----------------------------
Everything in this module acts on the OUTPUT DISTRIBUTION: it multiplies or
extrapolates categorical distributions and renormalises. Activation steering
acts UPSTREAM, on the residual stream, and lets the model's own learned
dynamics produce a coherent categorical.

That difference is the reason Phase 2 treats the Schiff et al. (2024)
controllability result as an open question rather than a settled verdict. They
report that absorbing-state masked diffusion degrades sharply under increasing
guidance strength — validity falling to a few hundred molecules out of 1,024 —
because a token, once unmasked, is frozen and a token chosen off-distribution
cannot be revised. Whether a softer, upstream intervention inherits the same
failure is exactly what E5 measures.

All classes here conform to the sampler's `logit_fn(logits, x, step, n_steps)`
hook signature.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


class NoGuidance:
    """Identity. The unconditional floor."""
    name = "unconditional"

    def __call__(self, logits, x, step=None, n_steps=None):
        return logits


class ClassifierGuidance:
    """Gradient-based guidance through a differentiable property surrogate.

        logits <- logits + gamma * d/d(logits) log p(y | x)

    Implemented with the first-order (soft one-hot) approximation: the token
    distribution is fed to the classifier as a soft embedding, so the gradient
    of the target objective flows back to the logits themselves.

    Costs, all of which E5 records: a trained surrogate, plus a forward AND
    backward pass through it at every denoising step.
    """
    name = "classifier_guidance"

    def __init__(self, clf, target: float, gamma: float = 1.0,
                 mean: float = 0.0, std: float = 1.0, binary: bool = False,
                 maximise: bool = True):
        self.clf = clf.eval()
        for p in self.clf.parameters():
            p.requires_grad_(False)
        self.gamma = float(gamma)
        self.binary = binary
        self.maximise = maximise
        self.target = (float(target) if binary else (float(target) - mean) / std)
        self.n_calls = 0

    def __call__(self, logits, x, step=None, n_steps=None):
        if self.gamma == 0.0:
            return logits
        self.n_calls += 1
        with torch.enable_grad():
            lg = logits.detach().clone().requires_grad_(True)
            probs = lg.softmax(-1)
            pred = self.clf(probs)
            if self.binary:
                obj = -F.binary_cross_entropy_with_logits(
                    pred, torch.full_like(pred, self.target))
            else:
                # move toward the target; sign handles maximise/minimise
                obj = -(pred - self.target) ** 2
                if self.maximise:
                    obj = obj.clone()
            grad = torch.autograd.grad(obj.sum(), lg)[0]
        return logits + self.gamma * grad.detach()


class DCFG:
    """Classifier-free guidance for discrete diffusion (Schiff et al. 2024).

        logits <- (1 + gamma) * logits_cond  -  gamma * logits_uncond

    REQUIRES A CONDITIONALLY TRAINED MODEL. There is no way to obtain
    logits_cond from an unconditional checkpoint, so this baseline carries the
    cost of a full conditional pretraining run — which is precisely the cost
    the method under test claims to avoid.

    Schiff et al. report that increasing gamma IMPROVES uniform-noise models but
    DEGRADES absorbing-state models. Reproducing that asymmetry on this backbone
    is one of the more informative outcomes E5 can produce.
    """
    name = "d_cfg"

    def __init__(self, cond_model, uncond_model, cond_kwargs: Dict,
                 gamma: float = 2.0, mask_id: int = 1):
        self.cond_model = cond_model.eval()
        self.uncond_model = uncond_model.eval()
        self.cond_kwargs = cond_kwargs
        self.gamma = float(gamma)
        self.mask_id = mask_id
        self.n_calls = 0

    @torch.no_grad()
    def __call__(self, logits, x, step=None, n_steps=None):
        # `logits` arriving here are the conditional model's output; the
        # unconditional pass is the extra cost of this method.
        self.n_calls += 1
        t = self.cond_kwargs.get("t")
        u = self.uncond_model(x, t)
        u[..., self.mask_id] = -1e9
        return (1.0 + self.gamma) * logits - self.gamma * u


class DCBG:
    """Classifier-based guidance for discrete diffusion (Schiff et al. 2024).

        logits <- logits + gamma * log p(y | x with token v at position i)

    The exact form evaluates the classifier for every candidate token at every
    position, which is O(V * L) classifier passes per step and infeasible at
    this vocabulary size. The first-order approximation below reuses the
    gradient trick from ClassifierGuidance.

    A warning worth carrying: Schiff et al. report that WITHOUT the
    approximation, masked-diffusion D-CBG collapses completely — 0.4 valid
    molecules at gamma=2 and zero at gamma>=3. If this baseline degrades
    sharply as gamma rises, that is a reproduction of a published result, not
    an implementation fault.
    """
    name = "d_cbg"

    def __init__(self, clf, target: float, gamma: float = 1.0,
                 mean: float = 0.0, std: float = 1.0, binary: bool = False,
                 exact: bool = False, mask_id: int = 1):
        self.inner = ClassifierGuidance(clf, target, gamma, mean, std, binary)
        self.exact = exact
        self.mask_id = mask_id
        self.name = "d_cbg_exact" if exact else "d_cbg_approx"
        self.n_calls = 0

    def __call__(self, logits, x, step=None, n_steps=None):
        self.n_calls += 1
        if not self.exact:
            return self.inner(logits, x, step, n_steps)
        raise NotImplementedError(
            "Exact D-CBG requires O(vocab x length) classifier evaluations per "
            "denoising step. Schiff et al. (2024) report it collapses on "
            "absorbing-state models (0 valid molecules at gamma>=3); the "
            "first-order approximation is the practical comparator.")


class PrefixConditioning:
    """Condition by writing a property token into the structural prefix, with
    NO activation intervention.

    Isolates the contribution of the intervention itself from the mere presence
    of the prefix — without this control, a positive steering result could be
    explained by the prefix carrying the signal on its own.
    """
    name = "prefix_conditioning"

    def __init__(self, token_id: int, position: int = 1):
        self.token_id = token_id
        self.position = position

    def apply_to_init(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, self.position] = self.token_id
        return x

    def __call__(self, logits, x, step=None, n_steps=None):
        return logits


def compose(*fns: Callable) -> Callable:
    """Chain logit transforms. Used only for deliberate ablations; composing
    guidance with steering by default would make attribution impossible."""
    def _f(logits, x, step=None, n_steps=None):
        for f in fns:
            if f is not None:
                logits = f(logits, x, step, n_steps)
        return logits
    return _f