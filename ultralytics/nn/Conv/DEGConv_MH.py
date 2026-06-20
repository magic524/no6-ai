######################################## DEGConv_MH — Multi-Head for Small Objects  ########################################
"""
DEGConv Multi-Head (MH-DEGConv) — parallel multi-scale patch heads for small object detection:
  Head 1 (1×1): Original resolution, position details
  Head 2 (2×2): Local texture (current default)
  Head 3 (3×3): Larger context
  Learned softmax fusion weights per channel.
"""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k2, C3k


def image2patches_ps(x, patch_size):
    """Patch with arbitrary patch_size. Returns patched x and pad_info."""
    if patch_size == 1:
        return x, (0, 0)
    b, c, h, w = x.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    x = einops.rearrange(x, "b c (hg h) (wg w) -> (hg wg b) c h w", hg=patch_size, wg=patch_size)
    return x, (pad_h, pad_w)


def patches2image_ps(x, pad_info, patch_size):
    """Reverse of image2patches_ps."""
    if patch_size == 1:
        return x
    pad_h, pad_w = pad_info
    x = einops.rearrange(x, "(hg wg b) c h w -> b c (hg h) (wg w)", hg=patch_size, wg=patch_size)
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :-pad_h, :-pad_w] if (pad_h > 0 and pad_w > 0) else \
            x[:, :, :-pad_h, :] if pad_h > 0 else \
                x[:, :, :, :-pad_w]
    return x


class EdgeConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.w_conv = nn.Conv2d(mid_channels, mid_channels, (1, kernel_size),
                                stride=1, padding=(0, kernel_size // 2), groups=mid_channels)
        self.h_conv = nn.Conv2d(mid_channels, mid_channels, (kernel_size, 1),
                                stride=1, padding=(kernel_size // 2, 0), groups=mid_channels)
        self.out_proj = nn.Conv2d(mid_channels * 2, out_channels, 1, bias=True)

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        return self.out_proj(torch.cat([x_w, x_h], dim=1))


def _safe_gn(num_channels):
    """GroupNorm with safe num_groups that always divides num_channels."""
    if num_channels < 8:
        return nn.Identity()
    groups = num_channels // 8
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DEGConv_SingleHead(nn.Module):
    """Single DEGConv head with configurable patch_size."""
    def __init__(self, in_dim, mid_dim, nbins=36, patch_size=2, cell_size=(8, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.nbins = nbins
        self.cell_size = cell_size
        self.cell_area = cell_size[0] * cell_size[1]

        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, mid_dim, kernel_size=1),
            nn.Conv2d(mid_dim, in_dim, kernel_size=3, padding=1, bias=False),
            _safe_gn(in_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.weight = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=mid_dim, out_channels=in_dim),
            _safe_gn(in_dim),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            _safe_gn(in_dim),
        )

        self.sigmoid = nn.Sigmoid()

    def get_hog_feature(self, x, input_dtype):
        x_mean = x.mean(dim=1, keepdim=True)
        b, _, h, w = x_mean.shape
        device = x_mean.device
        cell_h = min(self.cell_size[0], h)
        cell_w = min(self.cell_size[1], w)
        h_cells = max(1, h // cell_h)
        w_cells = max(1, w // cell_w)
        crop_h = h_cells * cell_h
        crop_w = w_cells * cell_w
        dirs_crop = x_mean[:, :, :crop_h, :crop_w].to(dtype=input_dtype)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=input_dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=input_dtype, device=device).view(1, 1, 3, 3)
        dx = F.conv2d(dirs_crop, sobel_x, padding=1)
        dy = F.conv2d(dirs_crop, sobel_y, padding=1)
        gradient_dir = torch.atan2(dy, dx + 1e-8).abs()
        dirs = gradient_dir.reshape(b, h_cells, cell_h, w_cells, cell_w)
        dirs = dirs.permute(0, 1, 3, 2, 4).reshape(b, h_cells, w_cells, -1)
        bin_width = torch.pi / self.nbins
        bin_indices = (dirs.to(torch.float32) / bin_width).floor().long().clamp(0, self.nbins - 1)
        bin_indices_flat = bin_indices.reshape(-1, dirs.shape[-1])
        weight = torch.zeros(bin_indices_flat.shape[0], self.nbins, dtype=input_dtype, device=device)
        weight.scatter_add_(1, bin_indices_flat, torch.ones_like(bin_indices_flat, dtype=input_dtype))
        weight = weight.reshape(b, h_cells, w_cells, self.nbins) / self.cell_area
        start = torch.pi / (2 * self.nbins)
        hog_bins = torch.linspace(start, torch.pi - start, self.nbins, dtype=input_dtype, device=device)
        hog_feature = hog_bins[None, None, None, :] * weight
        hog_feature = hog_feature.permute(0, 3, 1, 2)
        hog_feature = F.interpolate(hog_feature, size=(h, w), mode='nearest')
        return hog_feature

    def forward(self, x):
        input_dtype = x.dtype
        b, c, h, w = x.shape

        # Patch
        x_patched, pad_info = image2patches_ps(x, self.patch_size)

        # HOG
        x_hog = self.get_hog_feature(x_patched, input_dtype)
        x_hog = self.hog_feat(x_hog)

        # Gate
        x1 = self.sigmoid(self.weight(x_patched + x_hog))
        x2 = self.conv(x_patched)
        x_out = x1 * x2

        # Restore
        return patches2image_ps(x_out, pad_info, self.patch_size)


class DEGConv_MH(nn.Module):
    """Multi-Head DEGConv: 3 parallel patch scales with learned fusion."""
    def __init__(self, in_dim, out_dim, nbins=36, cell_size=(8, 8)):
        super().__init__()
        # Compress mid_dim to control parameter count
        mid_dim = max(16, in_dim // 3)

        # 3 heads with different patch sizes
        self.heads = nn.ModuleList([
            DEGConv_SingleHead(in_dim, mid_dim, nbins, ps, cell_size)
            for ps in [1, 2, 3]
        ])

        # Learned channel-wise fusion weights
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(in_dim * 3, in_dim, 1, bias=True),
            nn.Softmax(dim=1),
        )

        # Post-fusion (safe GroupNorm — divisible check)
        gn_groups = max(1, in_dim // 8)
        if in_dim % gn_groups != 0:
            gn_groups = 1  # fallback to LayerNorm-equivalent
        self.fuse_block = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=mid_dim, out_channels=in_dim),
            nn.GroupNorm(gn_groups, in_dim),
        )

        self.conv_1x1 = Conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = x

        # Parallel heads
        outputs = [h(x) for h in self.heads]  # 3 x (B, C, H, W)
        concat = torch.cat(outputs, dim=1)    # (B, 3C, H, W)

        # Channel-wise fusion weights
        weights = self.fusion_proj(concat)    # (B, C, H, W) softmax over C dim

        # Weighted sum
        fused = sum(weights[:, i:i+1] * outputs[i] for i in range(3))

        # Residual + fuse
        out = residual + self.fuse_block(fused)
        return self.conv_1x1(out)


class Bottleneck_DEGConv_MH(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = DEGConv_MH(c1, c1)
        self.cv2 = DEGConv_MH(c2, c2)


class C3k_DEGConv_MH(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DEGConv_MH(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_DEGConv_MH(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DEGConv_MH(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DEGConv_MH(self.c, self.c, shortcut)
            for _ in range(n))
