"""
2D VQ-VAE with 16x16 patch embedding and SWIN window attention.

Identical to the offset-attention version (vqvae_2d_offset.py) EXCEPT the
attention module: PatchOffsetAttention -> SwinWindowAttention. This isolates
the effect of the attention type; encoder/decoder, VQ, and the MRI->PET
distance-loss training are unchanged.

We place TWO Swin blocks back-to-back at the 8x8 latent stage: the first with a
regular window (shift=0), the second with a shifted window (shift=2), following
Swin's "two successive blocks" design so information crosses window borders.

CREDIT
------
- Window attention + relative position bias: Swin Transformer
  (Liu et al. 2021; microsoft/Swin-Transformer). See models/swin_attention.py.
- VQ-VAE quantizer: van den Oord et al. 2017 / taming-transformers pattern.
- Conv encoder/decoder skeleton: same as our offset-attention AE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_attention import SwinWindowAttention
from .vq import VectorQuantizer2D


class ResBlock2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder2D(nn.Module):
    """128 -> 8 (factor 16). Two Swin blocks (reg + shifted) at the 8x8 stage."""
    def __init__(self, in_ch=1, base=64, z_ch=4, window_size=4, num_heads=4):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down = nn.ModuleList([
            nn.Sequential(ResBlock2D(base),   nn.Conv2d(base,   base*2, 4, 2, 1)),  # 64
            nn.Sequential(ResBlock2D(base*2), nn.Conv2d(base*2, base*2, 4, 2, 1)),  # 32
            nn.Sequential(ResBlock2D(base*2), nn.Conv2d(base*2, base*4, 4, 2, 1)),  # 16
            nn.Sequential(ResBlock2D(base*4), nn.Conv2d(base*4, base*4, 4, 2, 1)),  # 8
        ])
        # regular window then shifted window (Swin two-block scheme)
        self.attn1 = SwinWindowAttention(base*4, window_size, num_heads, shift=0)
        self.attn2 = SwinWindowAttention(base*4, window_size, num_heads, shift=window_size // 2)
        self.out = nn.Sequential(
            ResBlock2D(base*4), nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.Conv2d(base*4, z_ch, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        for d in self.down:
            h = d(h)
        h = self.attn1(h)
        h = self.attn2(h)
        return self.out(h)


class Decoder2D(nn.Module):
    def __init__(self, out_ch=1, base=64, z_ch=4, window_size=4, num_heads=4):
        super().__init__()
        self.inp = nn.Conv2d(z_ch, base*4, 1)
        self.attn1 = SwinWindowAttention(base*4, window_size, num_heads, shift=0)
        self.attn2 = SwinWindowAttention(base*4, window_size, num_heads, shift=window_size // 2)
        self.up = nn.ModuleList([
            nn.Sequential(ResBlock2D(base*4), nn.ConvTranspose2d(base*4, base*4, 4, 2, 1)),  # 16
            nn.Sequential(ResBlock2D(base*4), nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)),  # 32
            nn.Sequential(ResBlock2D(base*2), nn.ConvTranspose2d(base*2, base*2, 4, 2, 1)),  # 64
            nn.Sequential(ResBlock2D(base*2), nn.ConvTranspose2d(base*2, base,   4, 2, 1)),  # 128
        ])
        self.out = nn.Sequential(
            ResBlock2D(base), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, out_ch, 3, padding=1), nn.Tanh(),
        )

    def forward(self, z):
        h = self.inp(z)
        h = self.attn1(h)
        h = self.attn2(h)
        for u in self.up:
            h = u(h)
        return self.out(h)


class VQVAE2DSwin(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, z_ch=4,
                 num_embeddings=1024, commitment_cost=0.25,
                 window_size=4, num_heads=4):
        super().__init__()
        self.encoder = Encoder2D(in_ch, base, z_ch, window_size, num_heads)
        self.vq = VectorQuantizer2D(num_embeddings, z_ch, commitment_cost)
        self.decoder = Decoder2D(out_ch, base, z_ch, window_size, num_heads)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, idx
