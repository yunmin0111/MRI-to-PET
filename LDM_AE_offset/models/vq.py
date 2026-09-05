"""
Vector Quantizer for VQ-VAE.

SOURCE / CREDIT
---------------
Standard VQ-VAE quantizer, following:
  - van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), 2017.
    https://arxiv.org/abs/1711.00937
  - Reference implementation pattern from the taming-transformers VQGAN
    (CompVis/taming-transformers, VectorQuantizer) and our own prior
    vqgan3d.py, adapted here to 2D (Conv2d latents).

Straight-through estimator is used so gradients flow back to the encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer2D(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=4, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: (B, D, H, W)  -> (B, H, W, D) -> (BHW, D)
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, self.embedding_dim)

        # L2 distance to codebook, pick nearest code
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        idx = d.argmin(1)
        z_q = self.embedding(idx).view(z_perm.shape)

        # VQ losses: codebook + commitment
        codebook_loss = F.mse_loss(z_q, z_perm.detach())
        commit_loss = F.mse_loss(z_q.detach(), z_perm)
        vq_loss = codebook_loss + self.commitment_cost * commit_loss

        # straight-through: copy gradients from z_q to z
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()   # back to (B, D, H, W)
        return z_q, vq_loss, idx.view(z.shape[0], -1)

    def encode_continuous(self, z):
        # BBDM/LDM later uses the CONTINUOUS encoder output (VQ bypassed at
        # inference), matching our prior finding. Provided for convenience.
        return z
