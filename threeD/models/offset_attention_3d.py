"""
3D PCT Offset-Attention over 4x4x4-patch tokens.

SOURCE / CREDIT
---------------
Offset-attention from PCT (Guo et al. 2020, https://arxiv.org/pdf/2012.09688;
code https://github.com/qinglew/PointCloudTransformer, module.py SA_Layer).
Formula:  F_out = LBR(F_in - F_sa) + F_in
Normalization: softmax on dim 1, then L1-norm on dim 2 (focuses attention on
high-difference regions). Unchanged from PCT - only the tokens differ.

TOKENS (this project)
---------------------
The latent (C, 32, 32, 32) is turned into 4x4x4-patch tokens by a stride-4 3D
conv (ViT-style patch embedding): 32 -> 8 per axis, giving 8*8*8 = 512 tokens,
each of dimension d_model. Offset-attention runs over these 512 tokens, then a
stride-4 transpose conv unembeds back to (C, 32, 32, 32).
"""
import torch
import torch.nn as nn


class OffsetAttention1D(nn.Module):
    """PCT offset-attention on tokens (B, C, N)."""
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight            # tied Q/K (PCT ref)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):                                   # x: (B, C, N)
        x_q = self.q_conv(x).permute(0, 2, 1)              # (B, N, C/4)
        x_k = self.k_conv(x)                               # (B, C/4, N)
        x_v = self.v_conv(x)                               # (B, C, N)
        energy = torch.bmm(x_q, x_k)                       # (B, N, N)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))  # L1 on dim2
        x_sa = torch.bmm(x_v, attention.permute(0, 2, 1))  # (B, C, N)
        x_off = self.act(self.after_norm(self.trans_conv(x - x_sa)))  # LBR(F_in - F_sa)
        return x + x_off                                   # + residual


class PatchOffsetAttention3D(nn.Module):
    """
    Patch-embed a 3D latent into 4^3 tokens, run PCT offset-attention, unembed.

    in_ch    : latent channels C
    patch    : patch size per axis (4 -> 4x4x4 blocks)
    d_model  : token dimension for attention
    """
    def __init__(self, in_ch, patch=4, d_model=256):
        super().__init__()
        self.embed = nn.Conv3d(in_ch, d_model, kernel_size=patch, stride=patch)   # 32->8
        self.oa = OffsetAttention1D(d_model)
        self.unembed = nn.ConvTranspose3d(d_model, in_ch, kernel_size=patch, stride=patch)  # 8->32

    def forward(self, x):                                   # x: (B, C, 32,32,32)
        t = self.embed(x)                                  # (B, d_model, 8,8,8)
        B, D, a, b, c = t.shape
        t = t.flatten(2)                                   # (B, d_model, 512)
        t = self.oa(t)                                     # offset-attention over 512 tokens
        t = t.view(B, D, a, b, c)                          # (B, d_model, 8,8,8)
        return self.unembed(t)                             # (B, C, 32,32,32)
