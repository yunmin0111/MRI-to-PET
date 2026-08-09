"""
MRI -> PET Latent BBDM (no condition).

Two encoder modes (only the MRI encoder differs):
  --mri_encoder independent : MRI-only VQGAN encoder  (Model 1, unaligned)
  --mri_encoder aligned     : align-trained MRI encoder (Model 2, aligned to PET latent)

Common:
  z_pet = PET_VQGAN.encode(pet)        # target
  z_mri = <mri_encoder>(mri)           # source
  BBDM learns  z_mri -> z_pet
  generation: z_gen -> PET_VQGAN.decode -> PET

Based on sd_aligned_bbdm.py, with the SDF condition removed entirely.
BBDM direction mapping vs the SDF code:
  old z_mri(target)  -> here z_pet(target)
  old z_sdf(source)  -> here z_mri(source)
UNet in_channels = z_t(4) + source(4) = 8  (was 9 with a 1-ch condition).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, nibabel as nib, os, sys, argparse
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqgan3d import VQGAN3D, Encoder3D
from light_unet import LightUNet3D
from datasets.dataset_align_mri2pet import AlignMri2PetDataset

device = torch.device('cuda')

# ----- paths -----
PET_VQGAN_CKPT = 'results/c2v/c2v_vqgan_pet/checkpoint/vqgan_last.pth'
MRI_VQGAN_CKPT = 'results/c2v/c2v_vqgan_mri/checkpoint/vqgan_last.pth'
ALIGN_ENC_CKPT = 'results/c2v/c2v_mri_align_pet/checkpoint/mri_enc_last.pth'
DATA_ROOT = '/data/yunmin0111/dataset_pet_c2v'


class DictToObj:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)


class LatentBBDM(nn.Module):
    """Same Brownian-bridge diffusion as sd_aligned_bbdm.py, condition removed."""
    def __init__(self):
        super().__init__()
        self.num_timesteps = 1000
        self.sample_step = 10
        self.eta = 1.0
        self.max_var = 1.0
        self.denoise_fn = LightUNet3D(
            image_size=32, in_channels=8, model_channels=64,   # 4(z_t)+4(source), no cond
            out_channels=4, num_res_blocks=2, channel_mult=(1,2,4))
        T = self.num_timesteps
        m_t = np.linspace(0.001, 0.999, T+1)
        var_t = 2.*(m_t - m_t**2)*self.max_var
        to_t = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer('m_t', to_t(m_t))
        self.register_buffer('variance_t', to_t(var_t))

    def forward(self, z_target, z_source):
        b = z_target.shape[0]
        t = torch.randint(1, self.num_timesteps+1, (b,), device=z_target.device).long()
        m = self.m_t[t].view(b,1,1,1,1)
        v = self.variance_t[t].view(b,1,1,1,1)
        n = torch.randn_like(z_target)
        z_t = (1.-m)*z_target + m*z_source + torch.sqrt(v)*n
        inp = torch.cat((z_t, z_source), dim=1)                # no condition
        obj = m*(z_source - z_target) + torch.sqrt(v)*n
        return F.l1_loss(obj, self.denoise_fn(inp, timesteps=t))

    @torch.no_grad()
    def sample(self, z_source):
        b, dev = z_source.shape[0], z_source.device
        z_t = z_source.clone()
        steps = list(reversed(list(np.linspace(0, self.num_timesteps, self.sample_step+1, dtype=int))))
        for i in range(len(steps)-1):
            cs, ns = steps[i], steps[i+1]
            t = torch.full((b,), cs, device=dev, dtype=torch.long)
            inp = torch.cat((z_t, z_source), dim=1)
            obj_pred = self.denoise_fn(inp, timesteps=t)
            z0r = torch.clamp(z_t - obj_pred, -50., 50.)
            if ns == 0:
                z_t = z0r
            else:
                mt, mnt = self.m_t[cs], self.m_t[ns]
                vt, vnt = self.variance_t[cs], self.variance_t[ns]
                s2 = torch.clamp((vt-vnt*(1.-mt)**2/((1.-mnt)**2+1e-8))*vnt/(vt+1e-8), min=0)
                z_t = (1.-mnt)*z0r + mnt*z_source + \
                      torch.sqrt(torch.clamp((vnt-s2)/(vt+1e-8),min=0))*(z_t-(1.-mt)*z0r-mt*z_source) + \
                      torch.sqrt(s2)*self.eta*torch.randn_like(z_t)
        return z_t


def build_loaders():
    cfg = DictToObj({'root': DATA_ROOT, 'input_size': 128, 'depth_size': 128})
    tr = DataLoader(AlignMri2PetDataset(cfg, stage='train'), batch_size=1, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    te = DataLoader(AlignMri2PetDataset(cfg, stage='test'), batch_size=1, shuffle=False)
    return tr, te


def load_pet_vqgan():
    m = VQGAN3D(in_channels=1, embedding_dim=4, num_embeddings=1024, base_channels=32)
    st = torch.load(PET_VQGAN_CKPT, map_location='cpu')
    m.encoder.load_state_dict(st['encoder']); m.decoder.load_state_dict(st['decoder']); m.vq.load_state_dict(st['vq'])
    m = m.to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def load_mri_encoder(kind):
    """independent: MRI-only VQGAN encoder | aligned: align-trained encoder."""
    enc = Encoder3D(in_channels=1, embedding_dim=4, base_channels=32).to(device)
    if kind == 'independent':
        st = torch.load(MRI_VQGAN_CKPT, map_location='cpu')
        enc.load_state_dict(st['encoder'])
        print(f"MRI encoder: INDEPENDENT (from {MRI_VQGAN_CKPT})")
    elif kind == 'aligned':
        st = torch.load(ALIGN_ENC_CKPT, map_location='cpu')
        enc.load_state_dict(st['mri_encoder'])
        print(f"MRI encoder: ALIGNED (from {ALIGN_ENC_CKPT}, epoch {st.get('epoch','?')})")
    else:
        raise ValueError(kind)
    enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mri_encoder', required=True, choices=['independent', 'aligned'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--tag', default=None, help='override model name/dir')
    a = ap.parse_args()

    tag = a.tag or ('c2v_bbdm_mri2pet_' + a.mri_encoder)
    ckpt_dir = f'results/c2v/{tag}/checkpoint'
    sample_dir = f'results/c2v/{tag}/samples'
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(sample_dir, exist_ok=True)

    pet_vqgan = load_pet_vqgan()
    mri_encoder = load_mri_encoder(a.mri_encoder)
    print(f"PET VQGAN: {sum(p.numel() for p in pet_vqgan.parameters())/1e6:.2f}M (frozen)")
    print(f"MRI Encoder: {sum(p.numel() for p in mri_encoder.parameters())/1e6:.2f}M (frozen)")

    train_loader, test_loader = build_loaders()

    # quick cross-check on a couple of test samples
    print("\n=== Latent sanity check ===")
    for i, batch in enumerate(test_loader):
        if i >= 2: break
        pet = batch[0][0].to(device)
        mri = batch[1][0][:, :1].to(device)
        with torch.no_grad():
            z_pet = pet_vqgan.encode(pet)
            z_mri = mri_encoder(mri)
            pet_self = pet_vqgan.decode(z_pet)
            pet_cross = pet_vqgan.decode(z_mri)   # MRI latent -> PET decoder
        sp = -10*np.log10(F.mse_loss(pet_self, pet).item()+1e-8)
        cp = -10*np.log10(F.mse_loss(pet_cross, pet).item()+1e-8)
        l2 = torch.sqrt(F.mse_loss(z_mri, z_pet)).item()
        print(f"  Sample {i}: pet_self={sp:.1f}dB, cross(MRIlatent->PETdec)={cp:.1f}dB, z L2={l2:.4f}")

    bbdm = LatentBBDM().to(device)
    print(f"\nBBDM: {sum(p.numel() for p in bbdm.parameters())/1e6:.2f}M")
    opt = Adam(bbdm.parameters(), lr=1e-4)

    # affine for saving samples (standard identity; orientation handled separately)
    for ep in range(a.epochs):
        bbdm.train(); el = 0
        pbar = tqdm(train_loader, desc=f"BBDM-{a.mri_encoder} [{ep+1}/{a.epochs}]")
        for batch in pbar:
            pet = batch[0][0].to(device)
            mri = batch[1][0][:, :1].to(device)
            with torch.no_grad():
                z_pet = pet_vqgan.encode(pet)     # target
                z_mri = mri_encoder(mri)          # source
            opt.zero_grad()
            loss = bbdm(z_pet, z_mri)             # z_mri -> z_pet, no condition
            loss.backward(); opt.step()
            el += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {ep+1}: loss={el/len(train_loader):.4f}")

        torch.save({'model': bbdm.state_dict(), 'epoch': ep+1}, f'{ckpt_dir}/last_model.pth')
        if (ep+1) % 20 == 0:
            torch.save({'model': bbdm.state_dict(), 'epoch': ep+1}, f'{ckpt_dir}/model_epoch_{ep+1}.pth')

        if (ep+1) % 10 == 0:
            bbdm.eval()
            for j, batch in enumerate(test_loader):
                if j >= 2: break
                pet = batch[0][0].to(device)
                mri = batch[1][0][:, :1].to(device)
                with torch.no_grad():
                    z_mri = mri_encoder(mri)
                    z_gen = bbdm.sample(z_mri)
                    pet_gen = pet_vqgan.decode(z_gen)
                psnr = -10*np.log10(F.mse_loss(pet_gen, pet).item()+1e-8)
                print(f"  [gen] Sample {j}: PSNR={psnr:.1f}dB")
                if j == 0:
                    nib.save(nib.Nifti1Image(pet_gen[0,0].cpu().numpy(), np.eye(4)), f'{sample_dir}/ep{ep+1}_syn.nii.gz')
                    nib.save(nib.Nifti1Image(pet[0,0].cpu().numpy(), np.eye(4)), f'{sample_dir}/ep{ep+1}_real.nii.gz')
    print("DONE")


if __name__ == '__main__':
    main()
