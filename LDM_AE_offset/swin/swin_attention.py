"""
Swin Transformer window attention for 2D feature-map tokens.

SOURCE / CREDIT
---------------
Window Multi-head Self-Attention (W-MSA) with relative position bias, from:
  - Paper : Liu et al., "Swin Transformer: Hierarchical Vision Transformer
            using Shifted Windows", ICCV 2021. https://arxiv.org/pdf/2103.14030
            (Eq. 4: Attention(Q,K,V) = SoftMax(QK^T/sqrt(d) + B) V, with
             relative position bias B)
  - Code  : official microsoft/Swin-Transformer, models/swin_transformer.py
            (class WindowAttention). Adapted here; the relative-position-bias
            table + index construction follows that reference closely.

ROLE IN THIS PROJECT
--------------------
Drop-in replacement for the PCT offset-attention we used before. The 2D VQ-VAE
is unchanged except that PatchOffsetAttention is swapped for this
SwinWindowAttention. Everything else (encoder/decoder, VQ, MRI->PET distance
loss) stays identical, so the two runs isolate the effect of the attention type.

Our latent grid is 8x8 (64 tokens). We use window_size=4 -> the 8x8 map splits
into 2x2 = 4 non-overlapping 4x4 windows, and attention is computed inside each
window (linear cost). A second block applies a shifted window (shift=2) so
information crosses window boundaries, as in Swin.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, ws):
    # x: (B, H, W, C) -> (num_windows*B, ws, ws, C)
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    # (num_windows*B, ws, ws, C) -> (B, H, W, C)
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """W-MSA with relative position bias (Swin Eq. 4). Operates on windows."""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size          # M
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table: (2M-1)*(2M-1) x nHeads
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1) * (2*window_size-1), num_heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # pairwise relative position index within a window (M*M, M*M)
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'))  # 2,M,M
        coords_flat = torch.flatten(coords, 1)                    # 2, M*M
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]   # 2, M*M, M*M
        rel = rel.permute(1, 2, 0).contiguous()                  # M*M, M*M, 2
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_index', rel.sum(-1))           # M*M, M*M

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (num_windows*B, N, C) with N = M*M
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # 3, Bw, nH, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)           # Bw, nH, N, N

        # add relative position bias B
        bias = self.rel_bias_table[self.rel_index.view(-1)].view(N, N, -1)
        bias = bias.permute(2, 0, 1).contiguous()               # nH, N, N
        attn = attn + bias.unsqueeze(0)

        attn = self.softmax(attn)
        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj(out)


class SwinWindowAttention(nn.Module):
    """
    Apply (optionally shifted) window attention to a (B, C, H, W) feature map.
    Same call signature as the previous PatchOffsetAttention, so it plugs into
    the same VQ-VAE unchanged.
    """
    def __init__(self, channels, window_size=4, num_heads=4, shift=0):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, window_size, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()                  # B, H, W, C
        shortcut = x
        x = self.norm(x)

        # cyclic shift (shifted-window block)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # partition into windows -> attention -> merge
        win = window_partition(x, self.window_size)             # nW*B, ws, ws, C
        win = win.view(-1, self.window_size * self.window_size, C)
        win = self.attn(win)
        win = win.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(win, self.window_size, H, W)         # B, H, W, C

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = shortcut + x                                        # residual
        return x.permute(0, 3, 1, 2).contiguous()              # B, C, H, W
