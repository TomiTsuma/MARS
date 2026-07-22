"""
Reverse sampling.

Ancestral sampling walks t: 1 -> 0 over N steps. At each step, every position
that is currently [MASK] is revealed with probability

    p = (alpha_s - alpha_t) / (1 - alpha_t)

and, if revealed, its token is drawn from the model's categorical. Revealed
tokens are then FROZEN for the rest of the trajectory — this is SUBS carry-over
constraint 2, and it is enforced here structurally.

A note that matters for Phase 2
-------------------------------
`confidence` unmasking (fill the highest-confidence positions first, LLaDA
style) usually gives better samples. But it makes the unmasking ORDER
data-dependent, which entangles the Phase 2 WHEN-axis steering results with the
model's own confidence dynamics. Keep `ancestral` canonical and report
`confidence` as a separate arm.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .objective import subs_logits
from .schedule import NoiseSchedule


@torch.no_grad()
def sample(model,
           schedule: NoiseSchedule,
           n_samples: int,
           seq_len: int,
           n_struct: int,
           bos_id: int,
           prefix_ids: List[int],
           mask_id: int,
           pad_id: int,
           n_steps: int = 128,
           temperature: float = 1.0,
           mode: str = "ancestral",
           device: str = "cuda",
           steer: Optional[Callable] = None,
           cache=None) -> torch.Tensor:
    """Generate `n_samples` sequences. Returns (n_samples, seq_len) int64.

    `steer` and `cache` are Phase 2 entry points. They are threaded through
    here so that Phase 2 requires no change to this file: steer(x, layer_idx,
    step_idx) is called inside each block, cache records activations.
    """
    model.eval()
    B, L = n_samples, seq_len

    # Initialise: structural prefix visible, everything else masked.
    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    x[:, 0] = bos_id
    for j, pid in enumerate(prefix_ids):
        x[:, 1 + j] = pid

    body = torch.zeros(L, dtype=torch.bool, device=device)
    body[n_struct:] = True                       # positions the model may fill

    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for k in range(n_steps):
        t = ts[k].expand(B)
        s = ts[k + 1].expand(B)

        if cache is not None:
            cache.set_step(k)
        logits = model(x, t, steer=steer, cache=cache)
        logits = subs_logits(logits, mask_id) / max(temperature, 1e-6)
        probs = logits.softmax(-1)

        is_masked = x.eq(mask_id) & body.unsqueeze(0)
        if not is_masked.any():
            break

        if mode == "ancestral":
            p_unmask = schedule.posterior_unmask_prob(ts[k], ts[k + 1])
            reveal = is_masked & (torch.rand_like(x, dtype=torch.float) < p_unmask)
        elif mode == "confidence":
            # reveal a fixed budget of the most confident masked positions
            conf = probs.max(-1).values.masked_fill(~is_masked, -1.0)
            budget = max(1, int(is_masked.sum(dim=1).float().mean().item()
                                / max(1, n_steps - k)))
            idx = conf.topk(budget, dim=1).indices
            reveal = torch.zeros_like(is_masked)
            reveal.scatter_(1, idx, True)
            reveal &= is_masked
        else:
            raise ValueError(mode)

        flat = probs.view(-1, probs.shape[-1])
        drawn = torch.multinomial(flat, 1).view(B, L)
        # CARRY-OVER: only masked-and-revealed positions change. Everything
        # already unmasked is untouched, by construction.
        x = torch.where(reveal, drawn, x)

    # Any position still masked at the end becomes pad (should be rare; log it).
    still = x.eq(mask_id)
    if still.any():
        x = torch.where(still, torch.full_like(x, pad_id), x)
    return x


@torch.no_grad()
def sample_to_smiles(model, schedule, tokenizer, n_samples: int,
                     n_steps: int = 128, device: str = "cuda",
                     batch_size: int = 512, **kw) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        x = sample(model, schedule, b, tokenizer.max_len, tokenizer.n_struct,
                   tokenizer.bos_id, tokenizer.prefix_ids, tokenizer.mask_id,
                   tokenizer.pad_id, n_steps=n_steps, device=device, **kw)
        out.extend(tokenizer.decode_smiles(row.tolist()) for row in x.cpu())
        remaining -= b
    return out
