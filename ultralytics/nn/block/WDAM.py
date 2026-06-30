######################################## CVPR2026 WDAM   by AI Little monster start ########################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class WDAM(nn.Module):
    """Wavelet-Driven Attention Module (CVPR2026)."""

    def __init__(self, dim, num_heads=8, window_size=5, shift_size=2, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Manual Haar DWT/IDWT filters (AMP-safe, no custom autograd)
        self._init_wavelet_filters()

        # 高频分支
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, dim, 1, bias=bias),
            nn.ReLU(inplace=True)
        )
        self.high_out = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True)
        )
        # 低频注意力QKV
        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def _init_wavelet_filters(self):
        """Haar wavelet filters as buffers (not parameters) for AMP-safe DWT/IDWT."""
        # Haar: low=[1,1]/sqrt(2), high=[1,-1]/sqrt(2) -- orthogonal
        h0 = torch.tensor([1.0, 1.0]) / (2 ** 0.5)  # low-pass
        h1 = torch.tensor([1.0, -1.0]) / (2 ** 0.5)  # high-pass

        # 2D separable filters (outer product)
        ll = (h0[:, None] * h0[None, :]).view(1, 1, 2, 2)  # LL
        lh = (h0[:, None] * h1[None, :]).view(1, 1, 2, 2)  # LH
        hl = (h1[:, None] * h0[None, :]).view(1, 1, 2, 2)  # HL
        hh = (h1[:, None] * h1[None, :]).view(1, 1, 2, 2)  # HH

        # DWT filters (same as forward)
        self.register_buffer('dwt_ll', ll)
        self.register_buffer('dwt_lh', lh)
        self.register_buffer('dwt_hl', hl)
        self.register_buffer('dwt_hh', hh)

        # IDWT filters (same kernels for orthogonal Haar)
        self.register_buffer('idwt_ll', ll)
        self.register_buffer('idwt_lh', lh)
        self.register_buffer('idwt_hl', hl)
        self.register_buffer('idwt_hh', hh)

    def _dwt(self, x):
        """Haar 2D DWT via F.conv2d -- fully AMP compatible."""
        C = x.shape[1]
        groups = C
        w_ll = self.dwt_ll.expand(C, -1, -1, -1)
        w_lh = self.dwt_lh.expand(C, -1, -1, -1)
        w_hl = self.dwt_hl.expand(C, -1, -1, -1)
        w_hh = self.dwt_hh.expand(C, -1, -1, -1)
        LL = F.conv2d(x, w_ll, stride=2, groups=groups)
        LH = F.conv2d(x, w_lh, stride=2, groups=groups)
        HL = F.conv2d(x, w_hl, stride=2, groups=groups)
        HH = F.conv2d(x, w_hh, stride=2, groups=groups)
        return LL, LH, HL, HH

    def _idwt(self, LL, LH, HL, HH):
        """Haar 2D IDWT via F.conv_transpose2d -- fully AMP compatible."""
        C = LL.shape[1]
        groups = C
        w_ll = self.idwt_ll.expand(C, -1, -1, -1)
        w_lh = self.idwt_lh.expand(C, -1, -1, -1)
        w_hl = self.idwt_hl.expand(C, -1, -1, -1)
        w_hh = self.idwt_hh.expand(C, -1, -1, -1)
        x = F.conv_transpose2d(LL, w_ll, stride=2, groups=groups)
        x = x + F.conv_transpose2d(LH, w_lh, stride=2, groups=groups)
        x = x + F.conv_transpose2d(HL, w_hl, stride=2, groups=groups)
        x = x + F.conv_transpose2d(HH, w_hh, stride=2, groups=groups)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        # ========== 修复1：DWT强制输入为偶数尺寸，补齐原图 ==========
        pad_hw = 0
        if H % 2 != 0 or W % 2 != 0:
            pad_hw = 1
            x_pad = F.pad(x, (0, W%2, 0, H%2), mode='constant', value=0)
        else:
            x_pad = x
        # DWT分解
        LL, LH, HL, HH = self._dwt(x_pad)
        # 高频融合权重
        filter_hv = self.high_conv(torch.cat([LH, HL], dim=1))
        # QKV
        qkv = self.qkv_dwconv(self.qkv(LL))
        q, k, v_inp = qkv.chunk(3, dim=1)
        v = v_inp * filter_hv + v_inp
        # ========== 修复2：q/k/v共用同一套padding参数 ==========
        ll_shifted = self.shift(LL, self.shift_size)
        win_q, pad_h, pad_w, llH, llW = self.window_partition(ll_shifted)
        win_k, _, _, _, _ = self.window_partition(ll_shifted)
        win_v, _, _, _, _ = self.window_partition(v)
        B_win, Cq, ws, _ = win_q.shape
        hd = Cq // self.num_heads
        q = win_q.view(B_win, self.num_heads, hd, ws*ws)
        k = win_k.view(B_win, self.num_heads, hd, ws*ws)
        v = win_v.view(B_win, self.num_heads, hd, ws*ws)
        attn_out = self.window_attn(q, k, v)
        attn_out = attn_out.view(B_win, Cq, ws, ws)
        ll_out = self.window_reverse(attn_out, pad_h, pad_w, llH, llW)
        ll_out = self.rev_shift(ll_out, self.shift_size)
        ll_out = self.project_out(ll_out)
        # 高频重建
        h_all = self.high_out(torch.cat([LH, HL, HH], dim=1))
        LH_new, HL_new, HH_new = h_all.chunk(3, dim=1)
        recon = self._idwt(ll_out, LH_new, HL_new, HH_new)
        # ========== 修复3：还原回原图原始尺寸，保证残差相加维度完全一致 ==========
        if pad_hw > 0:
            recon = recon[..., :H, :W]
        return recon

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        newH, newW = H + pad_h, W + pad_w
        x = x.view(B, C, newH//ws, ws, newW//ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = x.view(-1, C, ws, ws)
        return windows, pad_h, pad_w, H, W

    def window_reverse(self, windows, pad_h, pad_w, oriH, oriW):
        ws = self.window_size
        newH = oriH + pad_h
        newW = oriW + pad_w
        B = windows.shape[0] // ((newH // ws) * (newW // ws))
        C = windows.shape[1]
        x = windows.view(B, newH//ws, newW//ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, newH, newW)
        if pad_h or pad_w:
            x = x[..., :oriH, :oriW]
        return x

    def shift(self, x, s):
        if s > 0:
            x = torch.roll(x, shifts=(-s, -s), dims=(2, 3))
        return x

    def rev_shift(self, x, s):
        if s > 0:
            x = torch.roll(x, shifts=(s, s), dims=(2, 3))
        return x

    def window_attn(self, q, k, v):
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        attn = torch.matmul(q.transpose(-2, -1), k)
        N = self.window_size * self.window_size
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpb = rpb.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + rpb
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        return out


class C2PSA_WDAM(nn.Module):
    """C2PSA with Wavelet-Driven Attention Module (WDAM) blocks."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(WDAM(self.c) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
######################################## CVPR2026 WDAM   by AI Little monster end ########################################
