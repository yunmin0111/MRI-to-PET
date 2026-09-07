"""
3D UNet denoiser for latent diffusion (MRI->PET), two conditioning modes.

SOURCE / CREDIT
---------------
Structure follows CompVis/latent-diffusion (ldm/modules/diffusionmodules/
openaimodel.py, UNetModel) and OpenAI improved-diffusion: ResBlocks with
sinusoidal timestep embedding, down/up sampling, and attention at coarse
resolutions. Adapted to 3D latents and to inject a spatial condition z_mri.

CONDITIONING MODES
------------------
mode='concat' : z_mri is concatenated to the noisy latent on the channel axis
                at the input (LDM concat conditioning; in_channels = z + cond).
mode='adagn'  : z_mri is NOT concatenated. Instead each ResBlock's GroupNorm is
                modulated by (timestep emb) AND (a per-resblock projection of a
                pooled z_mri), i.e. AdaGN-style scale/shift from the condition.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock3D(nn.Module):
    """3D ResBlock with timestep (and optional condition) modulation via AdaGN."""
    def __init__(self, in_ch, out_ch, emb_dim, cond_dim=0):
        super().__init__()
        self.in_norm = nn.GroupNorm(8, in_ch)
        self.in_conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        # timestep projection -> scale & shift
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        # optional condition projection (AdaGN): pooled z_mri -> scale & shift
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.out_norm = nn.GroupNorm(8, out_ch)
        self.out_conv = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb, cond_vec=None):
        h = self.in_conv(F.silu(self.in_norm(x)))
        # timestep scale/shift
        s, b = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        h = self.out_norm(h) * (1 + s[:, :, None, None, None]) + b[:, :, None, None, None]
        # condition scale/shift (AdaGN) if provided
        if self.cond_dim > 0 and cond_vec is not None:
            cs, cb = self.cond_proj(cond_vec).chunk(2, dim=1)
            h = h * (1 + cs[:, :, None, None, None]) + cb[:, :, None, None, None]
        h = self.out_conv(F.silu(h))
        return h + self.skip(x)


class LDMUNet3D(nn.Module):
    """
    3D UNet for 32^3 latent. mode selects conditioning.
      concat: in_channels already includes the condition channels.
      adagn : condition is pooled to a vector and injected per ResBlock.
    """
    def __init__(self, in_channels=4, out_channels=4, base=64,
                 channel_mult=(1, 2, 4), cond_dim=0):
        super().__init__()
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.time_base = base
        chs = [base * m for m in channel_mult]
        self.in_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)
        # down
        self.downs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.downs.append(ResBlock3D(chs[i], chs[i], emb_dim, cond_dim))
            self.down_convs.append(nn.Conv3d(chs[i], chs[i+1], 4, 2, 1))
        # mid
        self.mid = ResBlock3D(chs[-1], chs[-1], emb_dim, cond_dim)
        # up
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.up_convs.append(nn.ConvTranspose3d(chs[i+1], chs[i], 4, 2, 1))
            self.ups.append(ResBlock3D(chs[i]*2, chs[i], emb_dim, cond_dim))  # *2 for skip
        self.out = nn.Sequential(nn.GroupNorm(8, chs[0]), nn.SiLU(),
                                 nn.Conv3d(chs[0], out_channels, 3, padding=1))

    def forward(self, x, t, cond_vec=None):
        emb = self.time_mlp(timestep_embedding(t, self.time_base))
        h = self.in_conv(x)
        skips = []
        for res, dc in zip(self.downs, self.down_convs):
            h = res(h, emb, cond_vec)
            skips.append(h)
            h = dc(h)
        h = self.mid(h, emb, cond_vec)
        for uc, res in zip(self.up_convs, self.ups):
            h = uc(h)
            h = torch.cat([h, skips.pop()], dim=1)   # U-Net skip
            h = res(h, emb, cond_vec)
        return self.out(h)
