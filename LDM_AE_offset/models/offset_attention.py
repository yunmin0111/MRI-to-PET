"""
PCT Offset-Attention (OA) for 2D image patch tokens.

SOURCE / CREDIT
---------------
Adapted from the Point Cloud Transformer (PCT):
  - Paper : Guo et al., "PCT: Point Cloud Transformer", 2020.
            https://arxiv.org/pdf/2012.09688  (Sec. 3.3 Offset-Attention)
  - Code  : https://github.com/qinglew/PointCloudTransformer  (module.py, SA_Layer)
            (which itself follows the official https://github.com/MenghaoGuo/PCT)

WHAT WE KEEP FROM PCT
---------------------
The offset-attention formula (Eq. 7 in the paper):
    F_out = OA(F_in) = LBR(F_in - F_sa) + F_in
where F_sa is the self-attention output. Instead of using the SA feature
directly, PCT feeds the *offset* (F_in - F_sa) into an LBR block and adds a
residual. PCT also normalizes the attention map differently from vanilla
Transformers: softmax over the first axis, then L1 normalization over the
second axis. This "sharpens" attention and suppresses noise
(paper Sec. 3.3). We keep exactly this.

WHAT WE CHANGE FOR IMAGES
-------------------------
PCT operates on point tokens of shape (B, C, N) where N = number of points.
We operate on *image patch tokens*: a 2D feature map (B, C, H, W) is flattened
to (B, C, N) with N = H*W (each spatial location = one token). This is the same
tensor layout PCT uses, so the attention code is unchanged - only the source of
the tokens differs (image patches instead of 3D points).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OffsetAttention(nn.Module):
    """
    PCT-style offset-attention over tokens of shape (B, C, N).

    channels : token feature dimension C
    The Q/K use a reduced dimension (C//4) as in the PCT reference code.
    """
    def __init__(self, channels):
        super().__init__()
        # Q and K projections use reduced channels (C//4), following PCT module.py.
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # Q and K share weights in the PCT reference (tie them for stability).
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        # LBR applied to the OFFSET (F_in - F_sa): Linear(1x1 conv)+BN+ReLU.
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, C, N)  -- N tokens, each with C channels
        x_q = self.q_conv(x).permute(0, 2, 1)      # (B, N, C/4)
        x_k = self.k_conv(x)                        # (B, C/4, N)
        x_v = self.v_conv(x)                        # (B, C, N)

        # attention map (B, N, N) = Q . K
        energy = torch.bmm(x_q, x_k)

        # --- PCT normalization: softmax on dim 1, then L1-norm on dim 2 ---
        # (paper Sec 3.3: "softmax on the first dimension and l1-norm for the
        #  second dimension"). This sharpens weights vs. vanilla 1/sqrt(d)+softmax.
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        # weighted sum of values -> self-attention feature F_sa  (B, C, N)
        x_sa = torch.bmm(x_v, attention.permute(0, 2, 1))

        # --- Offset-Attention: LBR(F_in - F_sa) + F_in  (Eq. 7) ---
        x_offset = x - x_sa
        x_offset = self.act(self.after_norm(self.trans_conv(x_offset)))
        x = x + x_offset
        return x


class PatchOffsetAttention(nn.Module):
    """
    Wrap OffsetAttention so it can be dropped into a 2D CNN.

    Input : (B, C, H, W) feature map (already downsampled by the encoder,
            so H, W are small -> each location is a 16x16-patch-level token).
    Output: (B, C, H, W) after offset-attention over the H*W tokens.
    """
    def __init__(self, channels):
        super().__init__()
        self.oa = OffsetAttention(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2)             # (B, C, H*W) = (B, C, N)
        tokens = self.oa(tokens)          # PCT offset-attention
        return tokens.view(B, C, H, W)    # back to feature map
