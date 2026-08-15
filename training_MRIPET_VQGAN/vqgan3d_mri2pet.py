"""
MRI->PET AutoEncoder (VQGAN) : encoder+decoder+VQ+discriminator all trained,
but the reconstruction TARGET is the paired PET, not the input MRI.

    MRI --Enc--> z --VQ--> z_q --Dec--> recon
    Loss = distance(recon, PET) + VQ + adversarial(+ perceptual)

Goal: pull the MRI latent strongly toward PET so the encoder produces a
PET-aligned z, which will later feed the BBDM as source.

Based on vqgan3d_pet.py; only the (input, target) split changes:
  input  = MRI   (fed to model)
  target = PET   (used in every loss: L_rec, perceptual, discriminator)

--dist l1|l2 chooses the reconstruction distance (adds an L2 term or swaps).
Uses dataset 'align_mri2pet': batch[0][0]=PET(target), batch[1][0]=MRI(source).
"""
import os, sys, yaml, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqgan3d import VQGAN3D
from perceptual_loss_3d import PerceptualLoss3D
from datasets.dataset_align_mri2pet import AlignMri2PetDataset


class DictToObj:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)


class Mri2PetVQGANRunner:
    def __init__(self, config, gpu_id=0):
        self.config = config
        self.device = torch.device(f'cuda:{gpu_id}')
        mn = config['model']['model_name']
        self.ckpt_dir = f'results/c2v/{mn}/checkpoint'
        os.makedirs(self.ckpt_dir, exist_ok=True)

        vq = config['model']['vqgan']
        self.model = VQGAN3D(
            in_channels=1,
            embedding_dim=vq.get('embedding_dim', 4),
            num_embeddings=vq.get('num_embeddings', 1024),
            base_channels=vq.get('base_channels', 32),
            commitment_cost=vq.get('commitment_cost', 0.25),
            disc_weight=vq.get('disc_weight', 0.5),
        ).to(self.device)

        self.perceptual = PerceptualLoss3D().to(self.device)
        self.perceptual_weight = vq.get('perceptual_weight', 0.01)
        self.dist = config['model'].get('dist', 'l1')     # 'l1' or 'l2'
        self.l2_weight = config['model'].get('l2_weight', 1.0)

        lr = vq.get('lr', 1e-4)
        ae_params = list(self.model.encoder.parameters()) + \
                    list(self.model.decoder.parameters()) + \
                    list(self.model.vq.parameters())
        self.ae_optimizer = Adam(ae_params, lr=lr)
        self.disc_optimizer = Adam(self.model.discriminator.parameters(), lr=lr)
        print(f"MRI->PET VQGAN | dist={self.dist} | perc_w={self.perceptual_weight}")

    def _loader(self, stage):
        ds = AlignMri2PetDataset(DictToObj(self.config['data']['dataset_config']), stage=stage)
        return DataLoader(ds, batch_size=1, shuffle=(stage=='train'),
                          num_workers=4, pin_memory=True, drop_last=(stage=='train'))

    def _split(self, batch):
        pet = batch[0][0].to(self.device)          # TARGET
        mri = batch[1][0][:, :1].to(self.device)   # INPUT
        return mri, pet

    def train(self, max_epochs=100):
        loader = self._loader('train')
        print(f"=== MRI->PET VQGAN, {max_epochs} epochs (target=PET) ===")
        for epoch in range(max_epochs):
            self.model.train()
            el = {'rec': 0, 'vq': 0, 'adv': 0, 'perc': 0, 'l2': 0}
            active = set()
            pbar = tqdm(loader, desc=f"MRI2PET-VQGAN [{epoch+1}/{max_epochs}]")
            for batch in pbar:
                mri, pet = self._split(batch)

                # --- Disc update: real=PET, fake=Dec(Enc(MRI)) ---
                recon, vq_loss, idx, _ = self.model(mri)   # INPUT = MRI
                active.update(idx[0].cpu().numpy().tolist())
                self.disc_optimizer.zero_grad()
                d_loss = self.model.compute_disc_loss(pet, recon)   # TARGET = PET
                d_loss.backward()
                self.disc_optimizer.step()

                # --- AE update: all losses vs PET ---
                recon, vq_loss, _, _ = self.model(mri)
                self.ae_optimizer.zero_grad()
                ae_loss, losses = self.model.compute_ae_loss(pet, recon, vq_loss)  # TARGET = PET
                perc_loss = self.perceptual(pet, recon)
                total = ae_loss + self.perceptual_weight * perc_loss
                l2_val = 0.0
                if self.dist == 'l2':
                    l2_term = F.mse_loss(recon, pet)
                    total = total + self.l2_weight * l2_term
                    l2_val = l2_term.item()
                total.backward()
                self.ae_optimizer.step()

                el['rec'] += losses['rec']; el['vq'] += losses['vq']
                el['adv'] += losses['adv']; el['perc'] += perc_loss.item(); el['l2'] += l2_val
                pbar.set_postfix(rec=f"{losses['rec']:.4f}", perc=f"{perc_loss.item():.4f}")

            n = len(loader)
            usage = len(active) / self.model.vq.num_embeddings * 100
            summ = {k: v/n for k, v in el.items()}
            print(f"Epoch {epoch+1}: " + ", ".join(f"{k}={v:.4f}" for k,v in summ.items()) + f", codebook={usage:.1f}%")

            if (epoch+1) % 5 == 0:
                vr, vp = self._validate()
                print(f"  val_rec(vs PET)={vr:.4f}, val_psnr={vp:.1f}dB")

            state = {
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'vq': self.model.vq.state_dict(),
                'discriminator': self.model.discriminator.state_dict(),
                'epoch': epoch+1,
            }
            torch.save(state, os.path.join(self.ckpt_dir, 'vqgan_last.pth'))
            if (epoch+1) % 20 == 0:
                torch.save(state, os.path.join(self.ckpt_dir, f'vqgan_epoch_{epoch+1}.pth'))

    def _validate(self):
        loader = self._loader('val')
        self.model.eval()
        tot, psnr = 0, 0
        with torch.no_grad():
            for batch in loader:
                mri, pet = self._split(batch)
                recon, _, _, _ = self.model(mri)
                tot += F.l1_loss(recon, pet).item()
                psnr += -10*np.log10(F.mse_loss(recon, pet).item()+1e-8)
        n = len(loader)
        return tot/n, psnr/n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--gpu_ids', default='0')
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    runner = Mri2PetVQGANRunner(cfg, int(a.gpu_ids.split(',')[0]))
    epochs = cfg['model']['vqgan'].get('epochs', 100)
    runner.train(max_epochs=epochs)


if __name__ == '__main__':
    main()
