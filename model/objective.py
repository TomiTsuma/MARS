"""
Forward masking process and the training objective.

Two structural constraints are enforced in code rather than learned. Both come
from the SUBS (substitution) parameterisation, and both are the difference
between a stable run and a mysteriously diverging one:

  1. ZERO MASK PROBABILITY. The model never assigns probability to [MASK].
     Implemented by setting that logit to -inf before the softmax.

  2. CARRY-OVER. Already-unmasked tokens pass through unchanged. Implemented
     structurally in the sampler (see sampler.py), not by hoping the model
     learns it.

The loss is a per-token NELBO: summed over masked positions, divided by the
number of VALID positions (real molecule tokens). Prefix, BOS and pad appear in
neither numerator nor denominator.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from model.schedule import NoiseSchedule, sample_t


NEG_INF = -1e9   # use a large finite value, not -inf: -inf * 0 = nan in bf16 paths



def forward_mask(ids: torch.Tensor,
                 maskable: torch.Tensor,
                 t: torch.Tensor,
                 schedule: NoiseSchedule,
                 mask_id: int,
                 generator: Optional[torch.Generator] = None):
    """Apply the forward process q(x_t | x_0).

    Each maskable token is independently replaced by [MASK] with probability
    1 - alpha_t. Structural and pad positions are never touched.

    Returns (x_t, was_masked) where was_masked is the boolean set of positions
    that contribute to the loss.
    """
    p = schedule.mask_prob(t).unsqueeze(-1)                    # (B, 1)
    u = torch.rand(ids.shape, device=ids.device, generator=generator)
    was_masked = (u < p) & maskable
    x_t = torch.where(was_masked, torch.full_like(ids, mask_id), ids)
    return x_t, was_masked


def subs_logits(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    """SUBS constraint 1: the model may not predict the absorbing state."""
    logits = logits.clone()
    logits[..., mask_id] = NEG_INF
    return logits


def nelbo_loss(logits: torch.Tensor,
               x0: torch.Tensor,
               was_masked: torch.Tensor,
               valid: torch.Tensor,
               t: torch.Tensor,
               schedule: NoiseSchedule,
               mask_id: int,
               pad_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Weighted masked cross-entropy = continuous-time NELBO.

        L = E_t [ w(t) * sum_{i masked} -log p_theta(x0_i | x_t) ] / |valid|

    Returns a dict so the trainer can log the unweighted CE separately — the
    weighted loss is dominated by small-t terms and is a poor progress signal
    on its own.
    """
    logits = subs_logits(logits, mask_id)
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)    # (B, L)

    nll = -tok_logp * was_masked                                 # (B, L)
    w = schedule.weight(t).unsqueeze(-1)                         # (B, 1)

    denom = valid.sum().clamp(min=1).float()
    loss = (w * nll).sum() / denom

    with torch.no_grad():
        n_masked = was_masked.sum().clamp(min=1).float()
        ce = nll.sum() / n_masked                                # unweighted, per masked token
        correct = (logits.argmax(-1) == x0)
        acc = (correct & was_masked).sum().float() / n_masked
        # Honest accuracy: [PAD] targets are trivially easy and dominate the
        # denominator (often 60-80% of positions). Track the non-pad number.
        real = was_masked & (x0 != pad_id) if pad_id is not None else was_masked
        n_real = real.sum().clamp(min=1).float()
        real_acc = (correct & real).sum().float() / n_real

    return {"loss": loss, "ce": ce, "acc": acc, "real_acc": real_acc,
            "mask_frac": was_masked.sum().float() / denom}


def training_step(model,
                  batch: Dict[str, torch.Tensor],
                  schedule: NoiseSchedule,
                  mask_id: int,
                  pad_id: Optional[int] = None,
                  t_eps: float = 1e-3,
                  low_discrepancy: bool = True) -> Dict[str, torch.Tensor]:
    """One full forward pass: sample t, noise, predict, score."""
    ids, maskable, valid = batch["ids"], batch["maskable"], batch["valid"]
    B = ids.shape[0]

    t = sample_t(B, ids.device, t_eps=t_eps, low_discrepancy=low_discrepancy)
    x_t, was_masked = forward_mask(ids, maskable, t, schedule, mask_id)

    logits = model(x_t, t)
    return nelbo_loss(logits, ids, was_masked, valid, t, schedule, mask_id, pad_id)
