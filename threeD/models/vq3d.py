"""3D voxel-wise Vector Quantizer. VQ-VAE (van den Oord 2017) pattern."""
import torch, torch.nn as nn, torch.nn.functional as F
class VectorQuantizer3D(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=4, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings=num_embeddings; self.embedding_dim=embedding_dim
        self.commitment_cost=commitment_cost
        self.embedding=nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
    def forward(self, z):                        # z: (B,C,D,H,W)
        z_perm=z.permute(0,2,3,4,1).contiguous() # (B,D,H,W,C)
        flat=z_perm.view(-1,self.embedding_dim)
        d=(flat.pow(2).sum(1,keepdim=True)-2*flat@self.embedding.weight.t()
           +self.embedding.weight.pow(2).sum(1))
        idx=d.argmin(1); z_q=self.embedding(idx).view(z_perm.shape)
        vq_loss=F.mse_loss(z_q,z_perm.detach())+self.commitment_cost*F.mse_loss(z_q.detach(),z_perm)
        z_q=z_perm+(z_q-z_perm).detach()
        z_q=z_q.permute(0,4,1,2,3).contiguous()  # (B,C,D,H,W)
        return z_q, vq_loss, idx.view(z.shape[0],-1)
