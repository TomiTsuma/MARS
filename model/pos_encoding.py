import torch

def rope_tables(seq_len: int, d_head: int, base: float, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv)                       # (L, d_head/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, H, L, Dh)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)