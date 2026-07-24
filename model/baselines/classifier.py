"""
Property classifiers for the guidance baselines.

Classifier guidance and D-CBG require a differentiable surrogate f(x) -> y.
Training one is part of the cost of those methods, and measuring that cost is
part of experiment E5: the central practical claim of activation steering is
that it needs no surrogate at all, and a cost claim must be measured rather
than asserted.

Two variants:

  ActivationClassifier -- reads the frozen model's residual stream. Cheap, and
      the natural comparator because it uses the same representation activation
      steering does.

  TokenClassifier -- reads a soft one-hot over tokens, so gradients flow back
      to the logits. Required for the first-order D-CBG approximation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class ActivationClassifier(nn.Module):
    """MLP over a pooled residual-stream vector -> scalar property."""

    def __init__(self, d_model: int, hidden: int = 256, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class TokenClassifier(nn.Module):
    """Embed a (soft) token distribution and predict the property.

    Accepts either hard ids or a probability simplex. The soft path is what
    makes D-CBG differentiable with respect to the logits: probs @ embedding
    gives a soft embedding whose gradient flows back through the softmax.
    """

    def __init__(self, vocab_size: int, d_emb: int = 128, hidden: int = 256,
                 max_len: int = 128, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.emb = nn.Embedding(vocab_size, d_emb)
        self.enc = nn.Sequential(
            nn.Linear(d_emb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x) -> torch.Tensor:
        if x.dtype in (torch.long, torch.int):
            e = self.emb(x)                       # (B, L, d)
        else:                                     # soft: (B, L, V)
            e = x @ self.emb.weight               # differentiable w.r.t. logits
        h = self.enc(e).mean(dim=1)               # mean-pool over positions
        return self.head(h).squeeze(-1)


# ------------------------------------------------------------------ training
def train_token_classifier(ids: torch.Tensor, y: np.ndarray, vocab_size: int,
                           max_len: int, epochs: int = 12, lr: float = 1e-3,
                           batch_size: int = 128, binary: bool = False,
                           device: str = "cuda", seed: int = 0,
                           verbose: bool = True) -> Dict:
    """Train a TokenClassifier and report held-out accuracy plus WALL-CLOCK.

    The wall-clock figure is not incidental. It is the cost that activation
    steering avoids entirely, and E5 reports it alongside control quality.
    """
    import time
    torch.manual_seed(seed)
    n = ids.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_tr = int(0.8 * n)
    tr, te = perm[:n_tr], perm[n_tr:]

    yt = torch.tensor(np.asarray(y, dtype=np.float32))
    mu, sd = float(yt[tr].mean()), float(yt[tr].std()) or 1.0
    yn = (yt - mu) / sd

    clf = TokenClassifier(vocab_size, max_len=max_len, binary=binary).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()

    t0 = time.time()
    for ep in range(epochs):
        clf.train()
        idx = tr[torch.randperm(len(tr))]
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            xb = ids[b].to(device)
            yb = (yt[b] if binary else yn[b]).to(device)
            opt.zero_grad()
            loss = lossf(clf(xb), yb)
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 4 == 0:
            print(f"      epoch {ep+1}/{epochs}  loss {loss.item():.4f}")
    train_seconds = time.time() - t0

    clf.eval()
    with torch.no_grad():
        pred = clf(ids[te].to(device)).cpu()
        tgt = (yt[te] if binary else yn[te])
        if binary:
            acc = float(((pred > 0).float() == tgt).float().mean())
            metric = {"accuracy": acc}
        else:
            ss_res = float(((pred - tgt) ** 2).sum())
            ss_tot = float(((tgt - tgt.mean()) ** 2).sum())
            metric = {"r2": 1.0 - ss_res / max(ss_tot, 1e-8)}

    if verbose:
        print(f"      held-out {metric}  |  train wall-clock {train_seconds:.1f}s")
    return {"model": clf, "mean": mu, "std": sd, "binary": binary,
            "metric": metric, "train_seconds": train_seconds}