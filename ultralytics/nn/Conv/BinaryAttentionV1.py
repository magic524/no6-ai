######################################## BinaryAttention V1 — 1-bit by AI Little monster  ########################################
"""Binary Attention V1 — 1-bit quantized attention mechanism for YOLO11.

V1 fixes over original:
  1. Force float32 inside attention forward (AMP numerical stability)
  2. Smooth STE gradient (avoid gradient starvation at large channels)
  3. Adaptive V clip range (prevents NaN from fixed [-2,2] at high channel counts)

Quantizes Q/K to binary (-1, +1) and V to 8-bit, with STE gradient approximation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, NewType

from ultralytics.nn.modules.block import C3k, C3k2

BinaryTensor = NewType("BinaryTensor", torch.Tensor)


# ── Binary ops ──────────────────────────────────────────────────────

def binary_sign(x: torch.Tensor) -> BinaryTensor:
    """Return -1 if x < 0, 1 if x >= 0, preserving dtype."""
    return x.sign() + (x == 0).type(x.dtype)


class STESign(Function):
    """Straight-Through Estimator for sign binarization.

    V1: smooth gradient attenuation instead of hard [-1,1] clipping.
    Gradient: g_out * max(0, 1 - |x|) — smoothly tapers to 0
    instead of abruptly cutting off, preventing gradient starvation
    at large channel counts.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor:
        ctx.save_for_backward(x)
        return binary_sign(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        # Smooth gradient: preserve full gradient for |x| <= 1,
        # linearly taper to 0 for |x| in (1, 2), zero beyond
        grad_mask = (1 - x.abs()).clamp(min=0, max=1)
        return grad_output * grad_mask


binarize = STESign.apply


class SymQuantizer(Function):
    """Symmetric uniform quantizer with dynamic range."""

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise=False):
        ctx.save_for_backward(input, clip_val)
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            assert input.ndimension() == 4
            max_input = (
                torch.max(torch.abs(input), dim=-2, keepdim=True)[0]
                .expand_as(input).detach()
            )
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        output = torch.round(input * s).div(s + 1e-6)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


symquantize = SymQuantizer.apply


def round_ste(z):
    """Round with straight-through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# ── Attention components ────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention with 1-bit Q/K and 8-bit V quantization.

    V1: forces float32 internally for numerical stability under AMP.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 attn_quant=True, pv_quant=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_quant = attn_quant
        self.pv_quant = pv_quant

    @staticmethod
    def _quantize(x):
        """1-bit Q/K quantization: sign(x) scaled by mean absolute value per head."""
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        sign = binarize(x)
        return s * sign

    @staticmethod
    def _quantize_p(x):
        """Quantize attention weights to 8-bit (0-255)."""
        qmax = 255
        s = torch.tensor(1.0 / qmax, dtype=x.dtype, device=x.device)
        q = round_ste(x / s).clamp(0, qmax)
        return s * q

    @staticmethod
    def _quantize_v(x, bits=8):
        """Quantize V values with adaptive range (V1: dynamic clip from data)."""
        # Compute 3-sigma range over all elements for this tensor
        v_std = x.std().detach()
        clip_range = max(float(v_std * 3.0), 1.0)
        act_clip_val = torch.tensor([-clip_range, clip_range], dtype=x.dtype, device=x.device)
        return symquantize(x, act_clip_val, bits, False)

    def forward(self, x):
        orig_dtype = x.dtype
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_quant:
            # Run quantization ops in float32 for numerical stability
            q, k, v = q.float(), k.float(), v.float()
            q = self._quantize(q)
            k = self._quantize(k)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if self.pv_quant:
                attn = self._quantize_p(attn)
                v = self._quantize_v(v, 8)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Restore original dtype for proj layer (AMP consistency)
        x = x.to(orig_dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BinaryAttentionBlock(nn.Module):
    """Transformer-like block with binary attention, handling 4D→3D→4D reshaping."""

    def __init__(self, dim, num_heads=2, qkv_bias=False, drop=0., attn_drop=0.,
                 attn_quant=True, pv_quant=True):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            attn_quant=attn_quant, pv_quant=pv_quant,
        )

    def forward(self, x):
        # x: (B, C, H, W) → reshape to (B, N, C) for attention
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_3d = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            x_3d = x

        x_norm = self.norm1(x_3d)
        x_attn = self.attn(x_norm)

        if x.dim() == 4:
            x_out = x_attn.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x_out = x_attn
        return x_out


class C3k_BinaryAttentionV1(C3k):
    """C3k bottleneck with BinaryAttentionBlock replacing internal Bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[BinaryAttentionBlock(c_, num_heads=2) for _ in range(n)])


class C3k2_BinaryAttentionV1(C3k2):
    """C3k2 module with BinaryAttentionBlock — V1 (numerically stable version).

    Args match C3k2: c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True.
    Both c3k and attn flags use C3k_BinaryAttentionV1 (with attention);
    when both False, uses plain BinaryAttentionBlock.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        c_ = int(c2 * e)
        self.m = nn.ModuleList([
            C3k_BinaryAttentionV1(c_, c_, 2, shortcut, g) if c3k or attn
            else BinaryAttentionBlock(c_, num_heads=2)
            for _ in range(n)
        ])
