"""
MRI Encoder aligned to PET VQGAN latent space (MRI->PET, task 1).

Mirror of sdf_encoder_align.py:
  Phase 1: PET-only VQGAN trained (done, frozen).
  Phase 2: train MRI Encoder only (this script)
      MRI -> Enc_MRI -> z_mri -> PET_VQGAN_Decoder(frozen) -> PET'
      Loss = |PET - PET'|   (only Enc_MRI updates)
  Phase 3: BBDM
      MRI -> Enc_MRI -> z_mri -> BBDM -> z_pet -> PET_VQGAN_Dec -> PET

Difference from the SDF version: MRI is ALREADY [-1,1], so NO source
re-normalization (the old ((sdf+10)/20) step is removed).
VQ is bypassed: the continuous encoder output is fed straight to the
frozen decoder, exactly as in the SDF alignment (BBDM uses continuous z).
"""
import os, sys, yaml, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, argparse
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqgan3d import VQGAN3D, Encoder3D


class MriEncoderAlignRunner:
    def __init__(self, config, gpu_id=0):
        self.config = config
        self.device = torch.device(f'cuda:{gpu_id}')
        mn = config['model']['model_name']
        self.ckpt_dir = f'results/c2v/{mn}/checkpoint'
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 1. PET-only VQGAN load (frozen!)
        vq_cfg = config['model'].get('vqgan', {})
        self.vqgan = VQGAN3D(
            in_channels=1,
            embedding_dim=vq_cfg.get('embedding_dim', 4),
            num_embeddings=vq_cfg.get('num_embeddings', 1024),
            base_channels=vq_cfg.get('base_channels', 32),
        ).to(self.device)
        vq_path = config['model']['vqgan_checkpoint']
        state = torch.load(vq_path, map_location='cpu')
        self.vqgan.encoder.load_state_dict(state['encoder'])
        self.vqgan.decoder.load_state_dict(state['decoder'])
        self.vqgan.vq.load_state_dict(state['vq'])
        print(f"Loaded PET VQGAN from {vq_path}")
        self.vqgan.eval()
        for p in self.vqgan.parameters():
            p.requires_grad = False

        # 2. MRI Encoder (trainable, only this)
        self.mri_encoder = Encoder3D(
            in_channels=1,
            embedding_dim=vq_cfg.get('embedding_dim', 4),
            base_channels=vq_cfg.get('base_channels', 32),
        ).to(self.device)
        vqgan_p = sum(p.numel() for p in self.vqgan.parameters())
        mri_p = sum(p.numel() for p in self.mri_encoder.parameters())
        print(f"PET VQGAN (frozen): {vqgan_p/1e6:.2f}M")
        print(f"MRI Encoder (trainable): {mri_p/1e6:.2f}M")
        self.optimizer = Adam(self.mri_encoder.parameters(), lr=1e-4)

    def _loader(self, stage):
        from datasets.dataset_align_mri2pet import AlignMri2PetDataset
        class DictToObj:
            def __init__(self, d):
                for k, v in d.items(): setattr(self, k, v)
        ds = AlignMri2PetDataset(DictToObj(self.config['data']['dataset_config']), stage=stage)
        return DataLoader(ds, batch_size=1, shuffle=(stage=='train'),
                          num_workers=4, pin_memory=True, drop_last=(stage=='train'))

    def _get_pair(self, batch):
        pet = batch[0][0].to(self.device)            # target (1,1,D,H,W)
        mri = batch[1][0][:, :1].to(self.device)     # source (1,1,D,H,W) via [:, :1]
        return pet, mri

    def train(self, max_epochs=100):
        loader = self._loader('train')
        print(f"=== MRI Encoder Alignment -> PET latent, {max_epochs} epochs ===")
        print("MRI -> Enc_MRI -> z_mri -> PET_VQGAN_Dec(frozen) -> PET'")
        print("Loss = |PET - PET'|, only Enc_MRI updates!")
        for epoch in range(max_epochs):
            self.mri_encoder.train()
            epoch_loss = 0; epoch_psnr = 0
            pbar = tqdm(loader, desc=f"Align [{epoch+1}/{max_epochs}]")
            for batch in pbar:
                pet, mri = self._get_pair(batch)
                z_mri = self.mri_encoder(mri)             # MRI -> PET latent
                pet_recon = self.vqgan.decoder(z_mri)     # frozen decoder, grad flows to encoder
                loss = F.l1_loss(pet_recon, pet)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                psnr = -10 * np.log10(F.mse_loss(pet_recon, pet).item() + 1e-8)
                epoch_loss += loss.item(); epoch_psnr += psnr
                pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.1f}")
            n = len(loader)
            print(f"Epoch {epoch+1}: loss={epoch_loss/n:.4f}, psnr={epoch_psnr/n:.1f}dB")
            if (epoch+1) % 5 == 0:
                vl, vp = self._validate()
                print(f"  val_loss={vl:.4f}, val_psnr={vp:.1f}dB")
                self._compare_latents()
            torch.save({'mri_encoder': self.mri_encoder.state_dict(), 'epoch': epoch+1},
                       os.path.join(self.ckpt_dir, 'mri_enc_last.pth'))
            if (epoch+1) % 20 == 0:
                torch.save({'mri_encoder': self.mri_encoder.state_dict(), 'epoch': epoch+1},
                           os.path.join(self.ckpt_dir, f'mri_enc_epoch_{epoch+1}.pth'))

    def _validate(self):
        loader = self._loader('val')
        self.mri_encoder.eval()
        tl, tp = 0, 0
        with torch.no_grad():
            for batch in loader:
                pet, mri = self._get_pair(batch)
                z_mri = self.mri_encoder(mri)
                pet_recon = self.vqgan.decoder(z_mri)
                tl += F.l1_loss(pet_recon, pet).item()
                tp += -10 * np.log10(F.mse_loss(pet_recon, pet).item() + 1e-8)
        n = len(loader)
        return tl/n, tp/n

    def _compare_latents(self):
        """z_mri(from MRI encoder) vs z_pet(from PET encoder): alignment quality."""
        loader = self._loader('val')
        self.mri_encoder.eval()
        dists = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 5: break
                pet, mri = self._get_pair(batch)
                z_mri = self.mri_encoder(mri)
                z_pet = self.vqgan.encode(pet)
                dists.append(F.mse_loss(z_mri, z_pet).item())
        print(f"  z_mri vs z_pet MSE: {np.mean(dists):.4f} (lower = better aligned)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--gpu_ids', default='0')
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    runner = MriEncoderAlignRunner(cfg, int(a.gpu_ids.split(',')[0]))
    epochs = cfg['model'].get('align_epochs', 100)
    runner.train(max_epochs=epochs)


if __name__ == '__main__':
    main()
