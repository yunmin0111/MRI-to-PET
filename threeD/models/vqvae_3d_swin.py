"""
3D VQ-VAE (MRI->PET) with SWIN window attention over 4x4x4-patch tokens.
Identical to vqvae_3d_offset.py except the attention module
(PatchOffsetAttention3D -> PatchSwinAttention3D). Isolates attention type.

CREDIT: Swin (Liu 2021; microsoft/Swin-Transformer, torchvision 3D variant).
VQ-VAE + conv skeleton same as the offset 3D version.
"""
import torch
import torch.nn as nn
from .swin_attention_3d import PatchSwinAttention3D
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
    def __init__(self, in_ch=1, base=32, z_ch=4, patch=4, d_model=256, window_size=4, num_heads=4):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, base, 3, padding=1)
        self.down = nn.ModuleList([
            nn.Sequential(ResBlock3D(base),   nn.Conv3d(base,   base*2, 4, 2, 1)),  # 64
            nn.Sequential(ResBlock3D(base*2), nn.Conv3d(base*2, base*4, 4, 2, 1)),  # 32
        ])
        self.pre = nn.Sequential(ResBlock3D(base*4), nn.GroupNorm(8, base*4), nn.SiLU(),
                                 nn.Conv3d(base*4, z_ch, 1))
        self.attn = PatchSwinAttention3D(z_ch, patch, d_model, window_size, num_heads)
    def forward(self, x):
        h = self.stem(x)
        for d in self.down: h = d(h)
        z = self.pre(h)
        return z + self.attn(z)


class Decoder3D(nn.Module):
    def __init__(self, out_ch=1, base=32, z_ch=4, patch=4, d_model=256, window_size=4, num_heads=4):
        super().__init__()
        self.attn = PatchSwinAttention3D(z_ch, patch, d_model, window_size, num_heads)
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
        for u in self.up: h = u(h)
        return self.out(h)


class VQVAE3DSwin(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32, z_ch=4,
                 num_embeddings=1024, commitment_cost=0.25, patch=4, d_model=256,
                 window_size=4, num_heads=4):
        super().__init__()
        self.encoder = Encoder3D(in_ch, base, z_ch, patch, d_model, window_size, num_heads)
        self.vq = VectorQuantizer3D(num_embeddings, z_ch, commitment_cost)
        self.decoder = Decoder3D(out_ch, base, z_ch, patch, d_model, window_size, num_heads)
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z)
        return self.decoder(z_q), vq_loss, idx
