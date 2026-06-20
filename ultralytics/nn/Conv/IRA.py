import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super(AttentionBlock, self).__init__()
        self._spatial_attention_conv = nn.Conv2d(2, dim, kernel_size=3, padding=1)
        # Channel attention MLP
        self._channel_attention_conv0 = nn.Conv2d(1, dim, kernel_size=1, padding=0)
        self._channel_attention_conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self._out_conv = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}.")

        # Spatial attention
        mean = torch.mean(x, dim=1, keepdim=True)  # Mean/Max on C axis
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([mean, max_val], dim=1)  # [B, 2, H, W]
        spatial_attention = self._spatial_attention_conv(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention) * x

        # Channel attention. Uses average pool (mean) only
        channel_attention = torch.relu(self._channel_attention_conv0(mean))
        channel_attention = self._channel_attention_conv1(channel_attention)
        channel_attention = torch.sigmoid(channel_attention) * x

        attention = torch.cat([spatial_attention, channel_attention], dim=1)
        attention = self._out_conv(attention)
        return x + attention


class BaseBlock(nn.Module):
    def __init__(self, channels: int):
        super(BaseBlock, self).__init__()
        self._conv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))  # ELU activation
        features = self._conv1(features)
        x = x + features
        features = F.elu(self._conv2(x))
        features = self._conv3(features)
        return x + features


class IRA(nn.Module):
    """Improved Receptive Attention block with DW conv + dual attention."""

    def __init__(self, inc, ouc):
        super().__init__()
        self.conv = nn.Conv2d(inc, ouc, kernel_size=3, padding=1)
        self.baseBlock = nn.Sequential(BaseBlock(ouc), BaseBlock(ouc))
        self.attn = AttentionBlock(ouc)

    def forward(self, x):
        x = self.conv(x)
        x = self.baseBlock(x)
        x = self.attn(x)
        return x


class C3k2_IRA(nn.Module):
    """C3k2 wrapper for IRA module."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(IRA(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))