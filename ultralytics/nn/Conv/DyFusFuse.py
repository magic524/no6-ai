"""
DyFusFuse — Dynamic Frequency-Spatial Fusion module for YOLO11 neck.

Adapted from EFSI-DETR's DyFusNet (arXiv 2601.18597):
  - Replaces plain Concat in neck with frequency-spatial guided fusion
  - DMSD: Dynamic Multi-resolution Spectral Decomposition (simulated low/mid/high bands)
  - SFCM: Spatial-Frequency Cooperative Modulation (multi-kernel + SE channel attention)
  - Channel split: 25% goes through frequency path, 75% passes through directly

Design:
  1. Concat input feature maps along channel dimension
  2. 1x1 Conv to project to out_channels
  3. Channel split: c_freq = out_channels * expand_ratio (default 0.25)
  4. Frequency path: DMSD -> SFCM
  5. Concat freq + direct -> 1x1 Conv fusion

Registration: same pattern as FAAFusion (neck fusion, NOT in base_modules/repeat_modules).
  In tasks.py: elif m is DyFusFuse: c1 = [ch[x] for x in f]; c2 = ...
"""

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.conv import Conv


class DMSD(nn.Module):
    """Dynamic Multi-resolution Spectral Decomposition.

    Decomposes features into low/mid/high frequency-inspired bands using lightweight spatial operators (no FFT/DWT),
    with content-adaptive weighting.
    """

    def __init__(self, channels):
        super().__init__()
        # Content-adaptive weights via GAP -> 1x1 -> softmax(3)
        self.weight_conv = nn.Conv2d(channels, 3, 1)

        # Low-frequency: AvgPool (low-pass)
        self.low = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # Mid-frequency: Identity (all-pass)
        # High-frequency: DWConv (learnable high-pass)
        self.high = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        # Content-adaptive weights: [B, 3, 1, 1]
        v = F.adaptive_avg_pool2d(x, 1)  # GAP
        w = F.softmax(self.weight_conv(v), dim=1)  # [B, 3, 1, 1]

        # Frequency bands
        low = self.low(x)
        mid = x
        high = self.high(x)

        # Weighted sum
        return w[:, 0:1] * low + w[:, 1:2] * mid + w[:, 2:3] * high


class SFCM(nn.Module):
    """Spatial-Frequency Cooperative Modulation.

    Multi-kernel spatial aggregation (1x1 + DW3x3 + DW5x5) followed by SE-style channel attention.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        # Multi-kernel spatial aggregation
        self.conv1x1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.dw3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)

        # SE channel attention
        c_mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, c_mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Multi-kernel aggregation
        z = self.conv1x1(x) + self.dw3x3(x) + self.dw5x5(x)
        # Channel attention
        beta = self.se(z)
        return z * beta


class DyFusFuse(nn.Module):
    """Dynamic Frequency-Spatial Fusion module.

    Replaces Concat in YOLO11 neck with frequency-spatial guided fusion. Takes multiple input feature maps, concatenates
    them, then applies DMSD + SFCM on a portion of channels.

    Args:
        in_channels: list of input channel counts (from each feature map)
        out_channels: output channel count
        expand_ratio: fraction of channels going through frequency path (default 0.25)
    """

    def __init__(self, in_channels, out_channels, expand_ratio=0.25):
        super().__init__()
        self.total_in = sum(in_channels)
        self.out_channels = out_channels
        self.c_freq = max(int(out_channels * expand_ratio), 8)
        self.c_direct = out_channels - self.c_freq

        # Project concatenated features to out_channels
        self.proj = Conv(self.total_in, out_channels, 1)

        # Frequency path
        self.dmsd = DMSD(self.c_freq)
        self.sfcm = SFCM(self.c_freq)

        # Final fusion
        self.fuse = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        # x is a list of tensors [B, C_i, H, W] (from YOLO _predict_once, same as Concat)
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1)  # [B, sum(C_i), H, W]
        x = self.proj(x)  # [B, out_channels, H, W]

        # Channel split
        x_freq, x_direct = torch.split(x, [self.c_freq, self.c_direct], dim=1)

        # Frequency path
        x_freq = self.sfcm(self.dmsd(x_freq))

        # Concat + fuse
        out = torch.cat([x_freq, x_direct], dim=1)
        return self.fuse(out)
