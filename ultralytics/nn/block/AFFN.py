# Ultralytics 🚀 AGPL-3.0 License
# AFFN (Adaptive Frequency Fusion Network) — CVPR2026
# C2PSA replacement with FFT-based self-correlation FFN
# https://github.com/AI-little-monster/AFFN

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2PSA, PSABlock


class AFFN(nn.Module):
    """Frequency-domain self-correlation FFN.
    Replaces the standard FFN inside PSABlock with FFT-based auto-correlation fusion.
    """
    def __init__(self, in_features, hidden_features, out_features, bias=False):
        super().__init__()
        self.dim = in_features

        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)

        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.project_in(x)
        original_dtype = x.dtype
        B, C, H, W = x.shape

        x_float = x.float()

        # Global FFT (no patch split)
        Xf = torch.fft.rfft2(x_float)

        # Auto-correlation power spectrum
        power = Xf * torch.conj(Xf)
        R = torch.fft.irfft2(power, s=(H, W))

        # Frequency + spatial domain fusion
        Xf_new = Xf + self.alpha.to(dtype=Xf.real.dtype) * power
        x_out = torch.fft.irfft2(Xf_new, s=(H, W))
        x_out = x_out + self.beta.to(dtype=R.dtype) * R

        # Restore dtype
        x = x_out.to(dtype=original_dtype)

        # Gated feed-forward
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class PSABlock_AFFN(PSABlock):
    """PSABlock with AFFN replacing the internal FFN."""
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.ffn = AFFN(c, c * 2, c)


class C2PSA_AFFN(C2PSA):
    """C2PSA with PSABlock_AFFN (FFT-based attention)."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(
            *(PSABlock_AFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        )
