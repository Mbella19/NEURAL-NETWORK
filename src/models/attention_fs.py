"""Frequency Spectrum Attention (FSAtten) - simplified Fourier gating for sequences."""
from __future__ import annotations

import torch
from torch import nn
import torch.fft


class FSAttention(nn.Module):
    """Applies a learnable frequency-domain gating followed by inverse transform."""

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, dim)
        freq = torch.fft.rfft(x, dim=1)
        mag = torch.abs(freq)
        gate = self.gate(mag)
        gated = freq * gate
        time = torch.fft.irfft(gated, n=x.size(1), dim=1)
        return self.out_proj(time)


__all__ = ["FSAttention"]
