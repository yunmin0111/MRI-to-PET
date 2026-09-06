"""
Train 3D VQ-VAE (MRI->PET) with either offset or swin attention.
--attn offset | swin   selects the model.
INPUT=MRI volume, TARGET=PET volume, LOSS=distance (--dist l1|l2) + VQ.
"""
import os, sys, argparse, yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.dataset_vol_mri2pet import VolMri2PetDataset


def psnr(mse):
    return 99.0 if mse <= 0 else 20*np.log10(2) - 10*np.log10(mse)


def build_model(attn, mc):
    kw = dict(in_ch=1, out_ch=1, base=mc.get('base_channels',32),
              z_ch=mc.get('embedding_dim',4), num_embeddings=mc.get('num_embeddings',1024),
              commitment_cost=mc.get('commitment_cost',0.25),
              patch=mc.get('patch',4), d_model=mc.get('d_model',256))
    if attn == 'offset':
        from models.vqvae_3d_offset import VQVAE3DOffset
        return VQVAE3DOffset(**kw)
    elif attn == 'swin':
        from models.vqvae_3d_swin import VQVAE3DSwin
        kw['window_size'] = mc.get('window_size', 4)
        kw['num_heads'] = mc.get('num_heads', 4)
        return VQVAE3DSwin(**kw)
    raise ValueError(attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--attn', required=True, choices=['offset','swin'])
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    device = torch.device(f'cuda:{a.gpu.split(",")[0]}')
    mc = cfg['model']; root = cfg['data']['root']
    dist = mc.get('dist','l1'); epochs = mc.get('epochs',100)
    bs = cfg['data'].get('batch_size', 1)   # 3D volumes are big -> small batch

    model = build_model(a.attn, mc).to(device)
    print(f"VQVAE3D-{a.attn} params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | dist={dist}")

    opt = torch.optim.Adam(model.parameters(), lr=mc.get('lr',1e-4))
    tr = DataLoader(VolMri2PetDataset(root,'train'), batch_size=bs, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    va = DataLoader(VolMri2PetDataset(root,'val'), batch_size=1, shuffle=False,
                    num_workers=2, pin_memory=True)

    ckpt_dir = f"results/{mc['model_name']}_{a.attn}/checkpoint"; os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(epochs):
        model.train(); run={'rec':0,'vq':0}; active=set()
        for mri, pet, _ in tr:
            mri, pet = mri.to(device), pet.to(device)
            recon, vq_loss, idx = model(mri)
            rec_loss = F.mse_loss(recon, pet) if dist=='l2' else F.l1_loss(recon, pet)
            loss = rec_loss + vq_loss
            opt.zero_grad(); loss.backward(); opt.step()
            run['rec']+=rec_loss.item(); run['vq']+=vq_loss.item()
            active.update(idx[0].detach().cpu().numpy().tolist())
        n=len(tr); usage=len(active)/model.vq.num_embeddings*100
        print(f"Epoch {ep+1}: rec={run['rec']/n:.4f}, vq={run['vq']/n:.4f}, codebook={usage:.1f}%")

        if (ep+1)%5==0:
            model.eval(); vr=vp=0; m=0
            with torch.no_grad():
                for mri, pet, _ in va:
                    mri, pet = mri.to(device), pet.to(device)
                    recon,_,_ = model(mri)
                    vr+=F.l1_loss(recon,pet).item(); vp+=psnr(F.mse_loss(recon,pet).item()); m+=1
            print(f"  val_rec(vs PET)={vr/m:.4f}, val_psnr={vp/m:.1f}dB")

        state={'encoder':model.encoder.state_dict(),'decoder':model.decoder.state_dict(),
               'vq':model.vq.state_dict(),'epoch':ep+1}
        torch.save(state, os.path.join(ckpt_dir,'ae_last.pth'))
        if (ep+1)%20==0:
            torch.save(state, os.path.join(ckpt_dir,f'ae_epoch_{ep+1}.pth'))
    print("DONE")

if __name__=='__main__':
    main()
