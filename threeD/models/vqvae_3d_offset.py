"""
3D VQ-VAE (MRI->PET) with PCT offset-attention over 4x4x4-patch tokens.

Pipeline:
  MRI (1,128,128,128)
   -> 3D conv encoder, 4x downsample: 128 -> 64 -> 32   => latent (C=z_ch*?,32^3)
      (we keep a modest channel width; attention runs on patch tokens)
   -> PatchOffsetAttention3D: 4^3 patch embed -> 512 tokens -> OA -> unembed
   -> VQ (voxel-wise over 32^3)
   -> 3D decoder, 4x upsample: 32 -> 64 -> 128
   -> PET (1,128,128,128)

CREDIT
------
- Offset-attention: PCT (Guo 2020; qinglew/PointCloudTransformer).
- VQ-VAE quantizer: van den Oord 2017 / taming pattern (3D).
- Conv encoder/decoder: standard 3D VQ-VAE skeleton for this MRI->PET task.
"""
import torch
import torch.nn as nn
from .offset_attention_3d import PatchOffsetAttention3D
from .vq3d import VectorQuantizer3D


class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)


class Encoder3D(nn.Module):
    """128 -> 32 (4x). latent channels = z_ch."""
    def __init__(self, in_ch=1, base=32, z_ch=4, patch=4, d_model=256):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, base, 3, padding=1)
        self.down = nn.ModuleList([
            nn.Sequential(ResBlock3D(base),   nn.Conv3d(base,   base*2, 4, 2, 1)),  # 64
            nn.Sequential(ResBlock3D(base*2), nn.Conv3d(base*2, base*4, 4, 2, 1)),  # 32
        ])
        self.pre = nn.Sequential(ResBlock3D(base*4), nn.GroupNorm(8, base*4), nn.SiLU(),
                                 nn.Conv3d(base*4, z_ch, 1))
        # offset-attention on 4^3 patch tokens of the z_ch latent
        self.attn = PatchOffsetAttention3D(z_ch, patch=patch, d_model=d_model)

    def forward(self, x):
        h = self.stem(x)
        for d in self.down:
            h = d(h)
        z = self.pre(h)              # (B, z_ch, 32,32,32)
        z = z + self.attn(z)         # residual offset-attention
        return z


class Decoder3D(nn.Module):
    """32 -> 128 (4x)."""
    def __init__(self, out_ch=1, base=32, z_ch=4, patch=4, d_model=256):
        super().__init__()
        self.attn = PatchOffsetAttention3D(z_ch, patch=patch, d_model=d_model)
        self.inp = nn.Conv3d(z_ch, base*4, 1)
        self.up = nn.ModuleList([
            nn.Sequential(ResBlock3D(base*4), nn.ConvTranspose3d(base*4, base*2, 4, 2, 1)),  # 64
            nn.Sequential(ResBlock3D(base*2), nn.ConvTranspose3d(base*2, base,   4, 2, 1)),  # 128
        ])
        self.out = nn.Sequential(ResBlock3D(base), nn.GroupNorm(8, base), nn.SiLU(),
                                 nn.Conv3d(base, out_ch, 3, padding=1), nn.Tanh())

    def forward(self, z):
        z = z + self.attn(z)
        h = self.inp(z)
        for u in self.up:
            h = u(h)
        return self.out(h)


class VQVAE3DOffset(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32, z_ch=4,
                 num_embeddings=1024, commitment_cost=0.25, patch=4, d_model=256):
        super().__init__()
        self.encoder = Encoder3D(in_ch, base, z_ch, patch, d_model)
        self.vq = VectorQuantizer3D(num_embeddings, z_ch, commitment_cost)
        self.decoder = Decoder3D(out_ch, base, z_ch, patch, d_model)
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, idx
