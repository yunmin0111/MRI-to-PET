"""
2D VQ-VAE with 16x16 patch embedding and PCT offset-attention.

DESIGN (per experiment spec)
----------------------------
- Works on 2D SLICES (a 3D 128^3 volume is processed slice-by-slice; the
  training loop flattens depth into the batch dimension).
- 16x16 patch embedding: a strided conv turns each 16x16 image patch into one
  token, so a 128x128 slice becomes an 8x8 = 64-token grid. This keeps the
  attention map small (64x64) and avoids OOM (per spec).
- PCT offset-attention (models/offset_attention.py) runs over those patch tokens.
- VQ-VAE quantizes the latent (models/vq.py).
- Encoder in = 1 (MRI slice), Decoder out = 1 (PET-like slice).

CREDIT
------
- Offset-attention: PCT (Guo et al. 2020; qinglew/PointCloudTransformer).
- VQ-VAE quantizer: van den Oord et al. 2017; taming-transformers pattern.
- Conv encoder/decoder skeleton: standard VQ-VAE/VQGAN 2D design, written here
  from scratch for the MRI->PET slice task.

The reconstruction TARGET is the paired PET slice, not the input MRI, so the
whole autoencoder learns MRI->PET translation with a plain distance loss
(set in the training script).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .offset_attention import PatchOffsetAttention
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
    """
    128x128 slice -> latent. Downsamples by 16 total (128 -> 8) so that each
    latent location corresponds to a 16x16 input patch (the '16x16 patch
    embedding'). Offset-attention is applied at the 8x8 token stage.
    """
    def __init__(self, in_ch=1, base=64, z_ch=4):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        # four stride-2 downsamples: 128 -> 64 -> 32 -> 16 -> 8
        self.down = nn.ModuleList([
            nn.Sequential(ResBlock2D(base),   nn.Conv2d(base,   base*2, 4, 2, 1)),  # 64
            nn.Sequential(ResBlock2D(base*2), nn.Conv2d(base*2, base*2, 4, 2, 1)),  # 32
            nn.Sequential(ResBlock2D(base*2), nn.Conv2d(base*2, base*4, 4, 2, 1)),  # 16
            nn.Sequential(ResBlock2D(base*4), nn.Conv2d(base*4, base*4, 4, 2, 1)),  # 8
        ])
        # PCT offset-attention over the 8x8 = 64 patch tokens
        self.attn = PatchOffsetAttention(base*4)
        self.out = nn.Sequential(
            ResBlock2D(base*4), nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.Conv2d(base*4, z_ch, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        for d in self.down:
            h = d(h)
        h = self.attn(h)          # offset-attention at 8x8
        return self.out(h)        # (B, z_ch, 8, 8)


class Decoder2D(nn.Module):
    """Mirror of the encoder: latent 8x8 -> 128x128 PET-like slice."""
    def __init__(self, out_ch=1, base=64, z_ch=4):
        super().__init__()
        self.inp = nn.Conv2d(z_ch, base*4, 1)
        self.attn = PatchOffsetAttention(base*4)
        self.up = nn.ModuleList([
            nn.Sequential(ResBlock2D(base*4), nn.ConvTranspose2d(base*4, base*4, 4, 2, 1)),  # 16
            nn.Sequential(ResBlock2D(base*4), nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)),  # 32
            nn.Sequential(ResBlock2D(base*2), nn.ConvTranspose2d(base*2, base*2, 4, 2, 1)),  # 64
            nn.Sequential(ResBlock2D(base*2), nn.ConvTranspose2d(base*2, base,   4, 2, 1)),  # 128
        ])
        self.out = nn.Sequential(
            ResBlock2D(base), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, out_ch, 3, padding=1), nn.Tanh(),   # output in [-1,1]
        )

    def forward(self, z):
        h = self.inp(z)
        h = self.attn(h)
        for u in self.up:
            h = u(h)
        return self.out(h)


class VQVAE2DOffset(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, z_ch=4,
                 num_embeddings=1024, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder2D(in_ch, base, z_ch)
        self.vq = VectorQuantizer2D(num_embeddings, z_ch, commitment_cost)
        self.decoder = Decoder2D(out_ch, base, z_ch)

    def encode(self, x):
        return self.encoder(x)            # continuous latent (for LDM later)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, idx
