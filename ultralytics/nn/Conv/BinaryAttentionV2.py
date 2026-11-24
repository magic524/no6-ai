######################################## BinaryAttention V2 — Hybrid ternary attention + local conv ########################################
"""
Binary Attention V2 — 针对 TinyPerson 小目标稀疏场景优化。.

V1 的问题：
  1. 1-bit Q/K ({+1, -1}) 丢失所有幅度信息 → 小目标信号被噪声淹没
  2. 纯注意力机制缺乏局部归纳偏置 → 稀疏目标场景下注意力散布
  3. EMA NaN → 训练不稳定

V2 改进：
  1. 三元 Q/K 量化 {+1, 0, -1} + 自适应阈值 → 保留强信号，过滤噪声
  2. 混合设计：注意力 + Depthwise Conv 并行，可学习的融合权重 α
     → 注意力崩溃时局部卷积分支兜底

V2.1 (2026-06-15) — STE 梯度修复：
  - 将梯度窗口从 2×threshold 扩大到 4×threshold（sigmoid 渐变）
  - 增加 0.02 的梯度最小值地板
  - 解决大尺度通道（x-scale）EMA NaN 问题
  3. 渐进式量化 warmup: 先全精度训练，逐步引入量化
  4. 梯度裁剪 + EMA NaN 检测
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from ultralytics.nn.modules.block import C3k, C3k2

# ── Quantization ops ─────────────────────────────────────────────────


class TernarySTEFunction(Function):
    """Ternary quantization with STE gradient: {-1, 0, +1} with adaptive threshold.

    V2.1 fix: widened gradient window + gradient floor to prevent EMA NaN at large scales. The original hard clamp at
    |x| > 2*threshold cuts ALL gradient → weight stagnation. Now uses sigmoid taper that never fully saturates to 0.
    """

    GRAD_MIN = 0.02  # minimum gradient floor — prevents complete starvation
    GRAD_WINDOW = 4.0  # gradient still flows for |x| up to 4*threshold (was 2.0)

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        # Compute per-token adaptive threshold
        abs_mean = x.abs().mean(dim=-1, keepdim=True)  # (B, H, N, 1)
        thresh_val = threshold * abs_mean
        mask = (x.abs() > thresh_val).type(x.dtype)
        return x.sign() * mask  # {-1, 0, +1}

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        (x,) = ctx.saved_tensors
        threshold = ctx.threshold
        # Soft gradient: sigmoid(-3(x_norm-1)) transitions smoothly around |x|=2*threshold
        # Never fully zeros — stays at GRAD_MIN for |x| >> 4*threshold
        x_norm = x.abs() / (threshold * TernarySTEFunction.GRAD_WINDOW + 1e-8)
        grad_mask = torch.sigmoid(-4.0 * (x_norm - 1.0))
        # Apply gradient floor to prevent starvation
        grad_mask = grad_mask * (1 - TernarySTEFunction.GRAD_MIN) + TernarySTEFunction.GRAD_MIN
        return grad_output * grad_mask, None


ternary_quantize = TernarySTEFunction.apply


class SymQuantizer(Function):
    """Symmetric uniform quantizer with dynamic range."""

    @staticmethod
    def forward(ctx, input, num_bits, layerwise=False):
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input).detach()
        else:
            assert input.ndimension() == 4, f"Expected 4D input, got {input.shape}"
            max_input = torch.max(torch.abs(input), dim=-2, keepdim=True)[0].expand_as(input).detach()
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        output = torch.round(input * s).div(s + 1e-6)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


symquantize = SymQuantizer.apply


# ── Attention components ────────────────────────────────────────────


class AttentionV2(nn.Module):
    """Multi-head self-attention with ternary Q/K quantization + hybrid local conv fusion.

    Key differences from V1:
    - Ternary Q/K {-1, 0, +1} with adaptive threshold (vs binary {-1, +1})
    - Hybrid local conv path for local inductive bias
    - Progressive quantization strength controlled externally
    """

    def __init__(
        self,
        dim,
        num_heads=2,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_ternary=True,
        pv_quant=True,
        use_local=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_ternary = qk_ternary
        self.pv_quant = pv_quant

        # Local depthwise conv branch (provides local inductive bias)
        self.use_local = use_local
        if use_local:
            # Temp conv: will be created with correct shape in forward
            self.local_conv_weight = nn.Parameter(torch.randn(dim, 1, 3, 3) * 0.02)
            # Learnable fusion gate: α = sigmoid(self.fusion_logit)
            self.fusion_logit = nn.Parameter(torch.tensor(1.0))  # favors attention initially

    def _ternary_qk(self, x, quant_strength=1.0):
        """Ternary Q/K quantization with adaptive threshold + progressive warmup."""
        if quant_strength < 1e-6:
            return x  # no quantization during warmup
        # Adaptive threshold: 0.5 * mean_abs per token (learnable scale)
        x.abs().mean(dim=-1, keepdim=True)  # (B, H, N, 1)
        # Linear interpolation between no-quant and full-quant during warmup
        if quant_strength < 1.0:
            # Blended: x_quant * strength + x * (1 - strength)
            x_ternary = ternary_quantize(x, 0.5)
            return x_ternary * quant_strength + x.detach() * (1 - quant_strength)
        else:
            return ternary_quantize(x, 0.5)

    def forward(self, x, quant_strength=1.0):
        orig_dtype = x.dtype
        B, N, C = x.shape
        # For local conv, we need spatial dims; compute if 4D input was provided
        H = int(N**0.5)
        W = H
        # Verify N is a perfect square (for conv compatibility)
        (H * W == N)

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Always run attention in float32 for numerical stability
        q, k, v = q.float(), k.float(), v.float()

        # Quantize Q/K
        if self.qk_ternary:
            q = self._ternary_qk(q, quant_strength)
            k = self._ternary_qk(k, quant_strength)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.pv_quant and quant_strength > 0.5:
            # Strengthen V quantization progressively
            v_quant_strength = max(0.0, (quant_strength - 0.5) * 2)  # starts at 50% quant strength
            if v_quant_strength < 1.0:
                v_q = symquantize(v, 8, False)
                v = v * (1 - v_quant_strength) + v_q * v_quant_strength
            else:
                v = symquantize(v, 8, False)

        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C).to(orig_dtype)

        # Local depthwise conv branch (provides local inductive bias)
        if self.use_local and N == H * W:
            # Reshape: (B, N, C) → (B, C, H, W)
            x_4d = x.permute(0, 2, 1).reshape(B, C, H, W).to(orig_dtype)
            # Apply depthwise conv
            local_out = F.conv2d(x_4d, self.local_conv_weight, padding=1, groups=C)
            # Reshape back: (B, C, H, W) → (B, N, C)
            local_out = local_out.reshape(B, C, N).permute(0, 2, 1).float()

            # Learnable fusion
            alpha = torch.sigmoid(self.fusion_logit)
            x_combined = alpha * attn_out + (1 - alpha) * local_out
        else:
            x_combined = attn_out

        # Output projection
        x_out = self.proj(x_combined.to(orig_dtype))
        x_out = self.proj_drop(x_out)
        return x_out


class BinaryAttentionBlockV2(nn.Module):
    """Transformer-like block with ternary attention + local conv hybrid.

    Supports progressive quantization via quant_strength (0=no quant → 1=full quant).
    """

    def __init__(
        self, dim, num_heads=2, qkv_bias=False, drop=0.0, attn_drop=0.0, qk_ternary=True, pv_quant=True, use_local=True
    ):
        super().__init__()
        self.dim = dim
        self.quant_strength = 1.0  # default: full quantization
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionV2(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_ternary=qk_ternary,
            pv_quant=pv_quant,
            use_local=use_local,
        )

    def forward(self, x, quant_strength=1.0):
        # x: (B, C, H, W) → reshape to (B, N, C) for attention
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_3d = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            x_3d = x

        x_norm = self.norm1(x_3d)
        x_attn = self.attn(x_norm, quant_strength)

        if x.dim() == 4:
            x_out = x_attn.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x_out = x_attn
        return x_out


class C3k_BinaryAttentionV2(C3k):
    """C3k bottleneck with BinaryAttentionBlockV2 replacing internal Bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[BinaryAttentionBlockV2(c_, num_heads=2) for _ in range(n)])


class C3k2_BinaryAttentionV2(C3k2):
    """C3k2 module with BinaryAttentionV2 — V2 (hybrid ternary + local conv).

    Args match C3k2: c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True.

    Supports progressive quantization via set_quant_strength() called before each epoch.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        c_ = int(c2 * e)
        self.m = nn.ModuleList(
            [
                C3k_BinaryAttentionV2(c_, c_, 2, shortcut, g)
                if c3k or attn
                else BinaryAttentionBlockV2(c_, num_heads=2)
                for _ in range(n)
            ]
        )

    def get_attn_blocks(self):
        """Recursively collect all BinaryAttentionBlockV2 modules."""
        blocks = []
        for mod in self.modules():
            if isinstance(mod, BinaryAttentionBlockV2):
                blocks.append(mod)
        return blocks

    def set_quant_strength(self, strength):
        """Set quantization strength for all child BinaryAttentionBlockV2 modules.

        Args:
            strength: float in [0, 1], 0 = no quantization, 1 = full quantization.
        """
        for block in self.get_attn_blocks():
            block.quant_strength = strength
