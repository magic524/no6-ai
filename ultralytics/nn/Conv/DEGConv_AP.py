######################################## DEGConv_AP — Adaptive Patch for Small Objects  ########################################
"""
DEGConv Adaptive Patch (AP-DEGConv) — improves small object detection by:
  1. Skipping 2×2 patching when feature map is small (H*W < 1600)
  2. Adaptive HOG cell_size based on spatial resolution
  3. Dilated EdgeConv for larger receptive field at deep layers
"""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k2, C3k


def image2patches_ap(x, patch_size=2):
    """Patch image with configurable patch_size. patch_size=1 means no patching."""
    if patch_size == 1:
        return x, (0, 0)
    b, c, h, w = x.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    hg, wg = patch_size, patch_size
    x = einops.rearrange(x, "b c (hg h) (wg w) -> (hg wg b) c h w", hg=hg, wg=wg)
    return x, (pad_h, pad_w)


def patches2image_ap(x, pad_info, patch_size=2):
    """Reverse of image2patches_ap."""
    if patch_size == 1:
        return x
    pad_h, pad_w = pad_info
    x = einops.rearrange(x, "(hg wg b) c h w -> b c (hg h) (wg w)", hg=patch_size, wg=patch_size)
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :-pad_h, :-pad_w] if (pad_h > 0 and pad_w > 0) else \
            x[:, :, :-pad_h, :] if pad_h > 0 else \
                x[:, :, :, :-pad_w]
    return x


class EdgeConvDilated(nn.Module):
    """EdgeConv with optional dilation for larger receptive field."""
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = kernel_size // 2 * dilation
        self.in_proj = nn.Conv2d(in_channels, mid_channels, 1, bias=True)
        self.w_conv = nn.Conv2d(mid_channels, mid_channels, (1, kernel_size),
                                stride=1, padding=(0, pad), dilation=(1, dilation),
                                groups=mid_channels)
        self.h_conv = nn.Conv2d(mid_channels, mid_channels, (kernel_size, 1),
                                stride=1, padding=(pad, 0), dilation=(dilation, 1),
                                groups=mid_channels)
        self.out_proj = nn.Conv2d(mid_channels * 2, out_channels, 1, bias=True)

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        return self.out_proj(x)


class DEGConv_AP(nn.Module):
    """Adaptive Patch DEGConv — patch_size adapts to spatial resolution."""
    def __init__(self, in_dim, out_dim, nbins=36, cell_size=(8, 8)):
        super().__init__()
        self.nbins = nbins
        self.cell_size = cell_size
        self.cell_area = cell_size[0] * cell_size[1]

        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.weight = nn.Sequential(
            EdgeConvDilated(in_channels=in_dim, mid_channels=in_dim // 2,
                           out_channels=in_dim, dilation=2 if in_dim >= 256 else 1),
            nn.GroupNorm(in_dim // 8, in_dim),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(in_dim // 8, in_dim),
        )

        self.fuse_block = nn.Sequential(
            EdgeConvDilated(in_channels=in_dim, mid_channels=in_dim // 2,
                          out_channels=in_dim, dilation=2 if in_dim >= 256 else 1),
            nn.GroupNorm(in_dim // 8, in_dim),
        )

        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = Conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def get_patch_size(self, h, w):
        """Determine patch size based on spatial dimensions."""
        spatial = h * w
        if spatial >= 1600:   # e.g., 40×40 or larger
            return 2          # use 2×2 patches
        else:                 # small feature maps, preserve spatial structure
            return 1          # no patching

    def get_hog_feature(self, x, input_dtype):
        x_mean = x.mean(dim=1, keepdim=True)
        b, _, h, w = x_mean.shape
        device = x_mean.device

        # Adaptive cell_size
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

        gradient_dir = torch.atan2(dy, dx + 1e-8)
        gradient_dir = torch.abs(gradient_dir)

        dirs = gradient_dir.reshape(b, h_cells, cell_h, w_cells, cell_w)
        dirs = dirs.permute(0, 1, 3, 2, 4).reshape(b, h_cells, w_cells, -1)

        bin_width = torch.pi / self.nbins
        bin_indices = (dirs.to(torch.float32) / bin_width).floor().long()
        bin_indices = torch.clamp(bin_indices, 0, self.nbins - 1)

        bin_indices_flat = bin_indices.reshape(-1, dirs.shape[-1])
        weight = torch.zeros(bin_indices_flat.shape[0], self.nbins,
                             dtype=input_dtype, device=device)
        weight.scatter_add_(1, bin_indices_flat,
                            torch.ones_like(bin_indices_flat, dtype=input_dtype))
        weight = weight.reshape(b, h_cells, w_cells, self.nbins) / self.cell_area

        start = torch.pi / (2 * self.nbins)
        hog_bins = torch.linspace(start, torch.pi - start, self.nbins,
                                  dtype=input_dtype, device=device)
        hog_feature = hog_bins[None, None, None, :] * weight
        hog_feature = hog_feature.permute(0, 3, 1, 2)
        hog_feature = F.interpolate(hog_feature, size=(h, w), mode='nearest')
        return hog_feature

    def forward(self, x):
        input_dtype = x.dtype
        residual = x
        b, c, h, w = x.shape
        patch_size = self.get_patch_size(h, w)

        # Patch (or not)
        x_patched, pad_info = image2patches_ap(x, patch_size)

        # HOG
        x_hog = self.get_hog_feature(x_patched, input_dtype)
        x_hog = self.hog_feat(x_hog)

        # Gate + conv
        x1 = self.sigmoid(self.weight(x_patched + x_hog))
        x2 = self.conv(x_patched)
        x_out = x1 * x2

        # Restore spatial
        x_out = patches2image_ap(x_out, pad_info, patch_size)

        # Residual + fuse
        x_out = x_out + residual
        x_out = self.fuse_block(x_out)
        return self.conv_1x1(x_out)


class Bottleneck_DEGConv_AP(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = DEGConv_AP(c1, c1)
        self.cv2 = DEGConv_AP(c2, c2)


class C3k_DEGConv_AP(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DEGConv_AP(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_DEGConv_AP(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DEGConv_AP(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DEGConv_AP(self.c, self.c, shortcut)
            for _ in range(n))
