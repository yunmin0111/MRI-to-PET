"""
MRI -> PET Latent BBDM with MONAI DiffusionModelUNet (ALIGNED encoder only).

Pipeline:
  MRI --[aligned MRI encoder (frozen)]--> z_mri  (source)
  PET --[PET VQGAN encoder (frozen)]----> z_pet  (target)
  Brownian-Bridge diffusion  z_mri -> z_pet, denoiser = MONAI DiffusionModelUNet
  generation: z_gen --[PET VQGAN decoder]--> PET

No condition. Only the aligned encoder path (per user's decision to run the
independent-encoder BBDM separately with light_unet if at all).

MONAI 1.5.2: from monai.networks.nets import DiffusionModelUNet
  call signature: unet(x, timesteps)  where x=(B,C,D,H,W), timesteps=(B,)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, nibabel as nib, os, sys, argparse
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqgan3d import VQGAN3D, Encoder3D
from datasets.dataset_align_mri2pet import AlignMri2PetDataset
from monai.networks.nets import DiffusionModelUNet

device = torch.device('cuda')

PET_VQGAN_CKPT = 'results/c2v/c2v_vqgan_pet/checkpoint/vqgan_last.pth'
ALIGN_ENC_CKPT = 'results/c2v/c2v_mri_align_pet/checkpoint/mri_enc_last.pth'
DATA_ROOT = '/data/yunmin0111/dataset_pet_c2v'


class DictToObj:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)


def build_monai_unet(latent_ch=4, base=64):
    """
    DiffusionModelUNet for 32^3 latent.
    in = z_t(latent_ch) + source(latent_ch); out = objective(latent_ch).
    Attention at the two coarser levels (16^3, 8^3) — cheap at 32^3 latent.
    Handles the channels/num_channels arg-name difference across MONAI versions.
    """
    common = dict(
        spatial_dims=3,
        in_channels=latent_ch*2,
        out_channels=latent_ch,
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=(0, base*2, base*4),
    )
    try:
        return DiffusionModelUNet(channels=(base, base*2, base*4), **common)
    except TypeError:
        return DiffusionModelUNet(num_channels=(base, base*2, base*4), **common)


class LatentBBDM(nn.Module):
    """Brownian-bridge diffusion, MONAI denoiser, no condition."""
    def __init__(self, latent_ch=4, base_ch=64):
        super().__init__()
        self.num_timesteps = 1000
        self.sample_step = 10
        self.eta = 1.0
        self.max_var = 1.0
        self.denoise_fn = build_monai_unet(latent_ch, base_ch)
        T = self.num_timesteps
        m_t = np.linspace(0.001, 0.999, T+1)
        var_t = 2.*(m_t - m_t**2)*self.max_var
        to_t = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer('m_t', to_t(m_t))
        self.register_buffer('variance_t', to_t(var_t))

    def _unet(self, z_t, z_source, t):
        inp = torch.cat((z_t, z_source), dim=1)     # (B, 2*ch, D,H,W)
        return self.denoise_fn(inp, timesteps=t)    # MONAI signature

    def forward(self, z_target, z_source):
        b = z_target.shape[0]
        t = torch.randint(1, self.num_timesteps+1, (b,), device=z_target.device).long()
        m = self.m_t[t].view(b,1,1,1,1)
        v = self.variance_t[t].view(b,1,1,1,1)
        n = torch.randn_like(z_target)
        z_t = (1.-m)*z_target + m*z_source + torch.sqrt(v)*n
        obj = m*(z_source - z_target) + torch.sqrt(v)*n
        pred = self._unet(z_t, z_source, t)
        return F.l1_loss(obj, pred)

    @torch.no_grad()
    def sample(self, z_source):
        b, dev = z_source.shape[0], z_source.device
        z_t = z_source.clone()
        steps = list(reversed(list(np.linspace(0, self.num_timesteps, self.sample_step+1, dtype=int))))
        for i in range(len(steps)-1):
            cs, ns = steps[i], steps[i+1]
            t = torch.full((b,), cs, device=dev, dtype=torch.long)
            obj_pred = self._unet(z_t, z_source, t)
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


def load_pet_vqgan():
    m = VQGAN3D(in_channels=1, embedding_dim=4, num_embeddings=1024, base_channels=32)
    st = torch.load(PET_VQGAN_CKPT, map_location='cpu')
    m.encoder.load_state_dict(st['encoder']); m.decoder.load_state_dict(st['decoder']); m.vq.load_state_dict(st['vq'])
    m = m.to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def load_aligned_encoder():
    enc = Encoder3D(in_channels=1, embedding_dim=4, base_channels=32).to(device)
    st = torch.load(ALIGN_ENC_CKPT, map_location='cpu')
    enc.load_state_dict(st['mri_encoder'])
    enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    print(f"Aligned MRI encoder loaded (epoch {st.get('epoch','?')})")
    return enc


def build_loaders():
    cfg = DictToObj({'root': DATA_ROOT, 'input_size': 128, 'depth_size': 128})
    tr = DataLoader(AlignMri2PetDataset(cfg, stage='train'), batch_size=1, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    te = DataLoader(AlignMri2PetDataset(cfg, stage='test'), batch_size=1, shuffle=False)
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--base_ch', type=int, default=64, help='MONAI UNet base channels')
    ap.add_argument('--tag', default='c2v_bbdm_mri2pet_aligned_monai')
    a = ap.parse_args()

    ckpt_dir = f'results/c2v/{a.tag}/checkpoint'
    sample_dir = f'results/c2v/{a.tag}/samples'
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(sample_dir, exist_ok=True)

    pet_vqgan = load_pet_vqgan()
    mri_encoder = load_aligned_encoder()
    print(f"PET VQGAN frozen: {sum(p.numel() for p in pet_vqgan.parameters())/1e6:.2f}M")
    print(f"Aligned encoder frozen: {sum(p.numel() for p in mri_encoder.parameters())/1e6:.2f}M")

    train_loader, test_loader = build_loaders()

    bbdm = LatentBBDM(latent_ch=4, base_ch=a.base_ch).to(device)
    print(f"BBDM (MONAI DiffusionModelUNet): {sum(p.numel() for p in bbdm.parameters())/1e6:.2f}M")
    opt = Adam(bbdm.parameters(), lr=1e-4)

    # sanity check
    print("\n=== Latent sanity check ===")
    for i, batch in enumerate(test_loader):
        if i >= 2: break
        pet = batch[0][0].to(device); mri = batch[1][0][:, :1].to(device)
        with torch.no_grad():
            z_pet = pet_vqgan.encode(pet); z_mri = mri_encoder(mri)
            pet_cross = pet_vqgan.decode(z_mri)
        cp = -10*np.log10(F.mse_loss(pet_cross, pet).item()+1e-8)
        l2 = torch.sqrt(F.mse_loss(z_mri, z_pet)).item()
        print(f"  Sample {i}: cross(MRIlatent->PETdec)={cp:.1f}dB, z L2={l2:.4f}, z shape={tuple(z_pet.shape)}")

    for ep in range(a.epochs):
        bbdm.train(); el = 0
        pbar = tqdm(train_loader, desc=f"BBDM-MONAI [{ep+1}/{a.epochs}]")
        for batch in pbar:
            pet = batch[0][0].to(device); mri = batch[1][0][:, :1].to(device)
            with torch.no_grad():
                z_pet = pet_vqgan.encode(pet)
                z_mri = mri_encoder(mri)
            opt.zero_grad()
            loss = bbdm(z_pet, z_mri)
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
                pet = batch[0][0].to(device); mri = batch[1][0][:, :1].to(device)
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
