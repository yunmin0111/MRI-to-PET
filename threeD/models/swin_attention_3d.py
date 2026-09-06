"""
3D Swin window attention over 4x4x4-patch tokens.

SOURCE / CREDIT
---------------
Window attention + 3D relative position bias from Swin Transformer
(Liu et al. 2021, https://arxiv.org/pdf/2103.14030; microsoft/Swin-Transformer,
and the 3D variant in torchvision.models.video.swin_transformer). Formula:
  Attention = SoftMax(QK^T/sqrt(d) + B) V , B = 3D relative position bias.

TOKENS
------
Latent (C,32,32,32) -> stride-4 patch embed -> 8^3 = 512 tokens (d_model).
The 8x8x8 token grid is split into windows of window_size^3; attention runs
inside each 3D window. A shifted-window block lets info cross window borders.
Then unembed back to (C,32,32,32).
"""
import torch
import torch.nn as nn


def window_partition_3d(x, ws):
    # x: (B, D,H,W, C) -> (nW*B, ws,ws,ws, C)
    B, D, H, W, C = x.shape
    x = x.view(B, D//ws, ws, H//ws, ws, W//ws, ws, C)
    return x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, ws, ws, ws, C)


def window_reverse_3d(win, ws, D, H, W):
    B = int(win.shape[0] / (D*H*W / ws**3))
    x = win.view(B, D//ws, H//ws, W//ws, ws, ws, ws, -1)
    return x.permute(0,1,4,2,5,3,6,7).contiguous().view(B, D, H, W, -1)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, ws, num_heads):
        super().__init__()
        self.ws=ws; self.num_heads=num_heads
        head_dim=dim//num_heads; self.scale=head_dim**-0.5
        self.rel_bias_table=nn.Parameter(torch.zeros((2*ws-1)**3, num_heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)
        coords=torch.stack(torch.meshgrid(torch.arange(ws),torch.arange(ws),torch.arange(ws),indexing='ij'))
        cf=torch.flatten(coords,1)                       # 3, ws^3
        rel=cf[:,:,None]-cf[:,None,:]                    # 3, N, N
        rel=rel.permute(1,2,0).contiguous()              # N,N,3
        rel[:,:,0]+=ws-1; rel[:,:,1]+=ws-1; rel[:,:,2]+=ws-1
        rel[:,:,0]*=(2*ws-1)*(2*ws-1); rel[:,:,1]*=(2*ws-1)
        self.register_buffer('rel_index', rel.sum(-1))   # N,N
        self.qkv=nn.Linear(dim,dim*3,bias=True); self.proj=nn.Linear(dim,dim)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self, x):                                 # (nW*B, N, C), N=ws^3
        Bw,N,C=x.shape
        qkv=self.qkv(x).reshape(Bw,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=(q*self.scale)@k.transpose(-2,-1)
        bias=self.rel_bias_table[self.rel_index.view(-1)].view(N,N,-1).permute(2,0,1)
        attn=attn+bias.unsqueeze(0)
        attn=self.softmax(attn)
        out=(attn@v).transpose(1,2).reshape(Bw,N,C)
        return self.proj(out)


class SwinBlock3D(nn.Module):
    def __init__(self, d_model, ws, num_heads, shift=0):
        super().__init__()
        self.ws=ws; self.shift=shift
        self.norm=nn.LayerNorm(d_model)
        self.attn=WindowAttention3D(d_model, ws, num_heads)
    def forward(self, x):                                 # (B, d_model, 8,8,8)
        B,C,D,H,W=x.shape
        x=x.permute(0,2,3,4,1).contiguous()              # B,D,H,W,C
        short=x; x=self.norm(x)
        if self.shift>0:
            x=torch.roll(x, (-self.shift,)*3, dims=(1,2,3))
        win=window_partition_3d(x, self.ws).view(-1, self.ws**3, C)
        win=self.attn(win).view(-1, self.ws,self.ws,self.ws, C)
        x=window_reverse_3d(win, self.ws, D,H,W)
        if self.shift>0:
            x=torch.roll(x, (self.shift,)*3, dims=(1,2,3))
        x=short+x
        return x.permute(0,4,1,2,3).contiguous()


class PatchSwinAttention3D(nn.Module):
    """Patch-embed 3D latent -> 8^3 tokens -> 2 swin blocks (reg+shift) -> unembed."""
    def __init__(self, in_ch, patch=4, d_model=256, window_size=4, num_heads=4):
        super().__init__()
        self.embed=nn.Conv3d(in_ch, d_model, patch, patch)       # 32->8
        self.b1=SwinBlock3D(d_model, window_size, num_heads, shift=0)
        self.b2=SwinBlock3D(d_model, window_size, num_heads, shift=window_size//2)
        self.unembed=nn.ConvTranspose3d(d_model, in_ch, patch, patch)  # 8->32
    def forward(self, x):
        t=self.embed(x)          # (B, d_model, 8,8,8)
        t=self.b1(t); t=self.b2(t)
        return self.unembed(t)
