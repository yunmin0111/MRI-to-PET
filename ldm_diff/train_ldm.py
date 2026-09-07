"""
Conditional Latent DDPM (MRI->PET), two conditioning modes.

Pipeline:
  PET --[PET VQGAN encoder, frozen]--> z_pet   (target latent, 32^3)
  MRI --[3D swin encoder, frozen]-----> z_mri   (condition latent, 32^3)
  DDPM learns to denoise z_pet, conditioned on z_mri.
  sample -> z_pet -> [PET VQGAN decoder] -> PET

--cond concat : UNet in_channels = 4(z_t) + 4(z_mri) = 8, cond concatenated
--cond adagn  : UNet in_channels = 4, z_mri pooled to a vector, injected via AdaGN

CREDIT: LDM (CompVis/latent-diffusion) for the latent-DDPM formulation and
concat conditioning; UNet from openaimodel pattern (see models/ldm_unet_3d.py).
Encoders: PET VQGAN (ours) + 3D swin AE encoder (ours, best MRI->PET AE).
"""
import os, sys, argparse, yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ldm_unet_3d import LDMUNet3D

device = torch.device('cuda')

# ---- frozen encoders/decoder ----
PET_VQGAN = '/data/yunmin0111/Cor2Vox/results/c2v/c2v_vqgan_pet/checkpoint/vqgan_last.pth'
SWIN_ENC  = '/data/yunmin0111/MRI-to-PET-LDM/threeD/results/ae_3d_swin/checkpoint/ae_last.pth'
DATA_ROOT = '/data/yunmin0111/dataset_pet_c2v'


def load_pet_vqgan():
    sys.path.insert(0, '/data/yunmin0111/Cor2Vox')
    from vqgan3d import VQGAN3D
    m = VQGAN3D(in_channels=1, embedding_dim=4, num_embeddings=1024, base_channels=32)
    st = torch.load(PET_VQGAN, map_location='cpu')
    m.encoder.load_state_dict(st['encoder']); m.decoder.load_state_dict(st['decoder']); m.vq.load_state_dict(st['vq'])
    m = m.to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def load_swin_encoder():
    sys.path.insert(0, '/data/yunmin0111/MRI-to-PET-LDM/threeD')
    from models.vqvae_3d_swin import Encoder3D
    enc = Encoder3D(in_ch=1, base=32, z_ch=4, patch=4, d_model=256, window_size=4, num_heads=4).to(device)
    st = torch.load(SWIN_ENC, map_location='cpu')
    enc.load_state_dict(st['encoder'])
    enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    return enc


class VolDS(torch.utils.data.Dataset):
    def __init__(self, root, stage):
        import glob, nibabel as nib
        sfx = {'train':'Tr','val':'Val','test':'Ts'}[stage]
        self.md = os.path.join(root, f'mri{sfx}'); self.pd = os.path.join(root, f'pet{sfx}')
        self.ids = [os.path.basename(f)[:-7] for f in sorted(glob.glob(os.path.join(self.pd,'*.nii.gz')))
                    if os.path.exists(os.path.join(self.md, os.path.basename(f)))]
        self.nib = nib
        print(f"[VolDS] {stage}: {len(self.ids)}")
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        pid=self.ids[i]
        mri=np.asarray(self.nib.load(os.path.join(self.md,f'{pid}.nii.gz')).get_fdata(),dtype=np.float32)
        pet=np.asarray(self.nib.load(os.path.join(self.pd,f'{pid}.nii.gz')).get_fdata(),dtype=np.float32)
        return torch.from_numpy(mri)[None], torch.from_numpy(pet)[None], pid


class DDPM:
    """Standard DDPM schedule (linear beta), epsilon-prediction."""
    def __init__(self, T=1000, device='cuda'):
        self.T=T
        beta=torch.linspace(1e-4,0.02,T,device=device)
        alpha=1-beta; abar=torch.cumprod(alpha,0)
        self.beta=beta; self.abar=abar
        self.sqrt_abar=torch.sqrt(abar); self.sqrt_1mabar=torch.sqrt(1-abar)
    def q_sample(self,z0,t,noise):
        return self.sqrt_abar[t][:,None,None,None,None]*z0 + self.sqrt_1mabar[t][:,None,None,None,None]*noise
    @torch.no_grad()
    def sample(self,unet,z_mri,shape,cond_mode):
        z=torch.randn(shape,device=device)
        for i in reversed(range(self.T)):
            t=torch.full((shape[0],),i,device=device,dtype=torch.long)
            if cond_mode=='concat':
                inp=torch.cat([z,z_mri],dim=1); eps=unet(inp,t)
            else:
                cvec=z_mri.mean(dim=(2,3,4)); eps=unet(z,t,cvec)
            a=1-self.beta[i]; abar=self.abar[i]
            z0=(z-self.sqrt_1mabar[i]*eps)/self.sqrt_abar[i]
            if i>0:
                noise=torch.randn_like(z)
                z=torch.sqrt(a)*(z-self.beta[i]/self.sqrt_1mabar[i]*eps)+torch.sqrt(self.beta[i])*noise
            else:
                z=z0
        return z


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cond',required=True,choices=['concat','adagn'])
    ap.add_argument('--epochs',type=int,default=200)
    ap.add_argument('--tag',default=None)
    a=ap.parse_args()
    tag=a.tag or f'ldm_ddpm_{a.cond}'
    ckpt_dir=f'results/{tag}/checkpoint'; os.makedirs(ckpt_dir,exist_ok=True)

    pet_vqgan=load_pet_vqgan(); swin_enc=load_swin_encoder()
    if a.cond=='concat':
        unet=LDMUNet3D(in_channels=8, out_channels=4, base=64, channel_mult=(1,2,4), cond_dim=0).to(device)
    else:
        unet=LDMUNet3D(in_channels=4, out_channels=4, base=64, channel_mult=(1,2,4), cond_dim=4).to(device)
    print(f"cond={a.cond} | UNet params {sum(p.numel() for p in unet.parameters())/1e6:.2f}M")

    ddpm=DDPM(T=1000,device=device)
    opt=torch.optim.Adam(unet.parameters(),lr=1e-4)
    tr=DataLoader(VolDS(DATA_ROOT,'train'),batch_size=1,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    te=DataLoader(VolDS(DATA_ROOT,'test'),batch_size=1,shuffle=False)

    for ep in range(a.epochs):
        unet.train(); el=0
        for mri,pet,_ in tr:
            mri,pet=mri.to(device),pet.to(device)
            with torch.no_grad():
                z_pet=pet_vqgan.encode(pet)        # target
                z_mri=swin_enc(mri)                # condition
            t=torch.randint(0,ddpm.T,(z_pet.shape[0],),device=device)
            noise=torch.randn_like(z_pet)
            z_t=ddpm.q_sample(z_pet,t,noise)
            if a.cond=='concat':
                inp=torch.cat([z_t,z_mri],dim=1); pred=unet(inp,t)
            else:
                cvec=z_mri.mean(dim=(2,3,4)); pred=unet(z_t,t,cvec)
            loss=F.mse_loss(pred,noise)            # epsilon prediction
            opt.zero_grad(); loss.backward(); opt.step()
            el+=loss.item()
        print(f"Epoch {ep+1}: loss={el/len(tr):.4f}")
        torch.save({'unet':unet.state_dict(),'epoch':ep+1},f'{ckpt_dir}/last.pth')
        if (ep+1)%20==0:
            torch.save({'unet':unet.state_dict(),'epoch':ep+1},f'{ckpt_dir}/ep{ep+1}.pth')
        if (ep+1)%10==0:
            unet.eval()
            for j,(mri,pet,_) in enumerate(te):
                if j>=2: break
                mri,pet=mri.to(device),pet.to(device)
                with torch.no_grad():
                    z_mri=swin_enc(mri)
                    z_gen=ddpm.sample(unet,z_mri,z_mri.shape,a.cond)
                    pet_gen=pet_vqgan.decode(z_gen)
                ps=-10*np.log10(F.mse_loss(pet_gen,pet).item()+1e-8)
                print(f"  [gen] {j}: PSNR={ps:.1f}dB")
    print("DONE")

if __name__=='__main__':
    main()
