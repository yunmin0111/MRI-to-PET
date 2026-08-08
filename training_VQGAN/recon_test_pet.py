#!/usr/bin/env python3
"""
Reconstruct held-out TEST PET volumes through the trained PET VQGAN.
Saves original + reconstruction side by side and reports PSNR/SSIM.

Model wiring follows vqgan3d_pet.py:
    from vqgan3d import VQGAN3D
    checkpoint has separate encoder/decoder/vq/discriminator state_dicts
    forward: x_recon, vq_loss, idx, _ = model(x)
"""
import os, sys, glob, yaml, argparse
import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, '/data/yunmin0111/Cor2Vox')
from vqgan3d import VQGAN3D


def psnr(a, b, data_range=2.0):   # data in [-1,1] -> range 2
    mse = np.mean((a - b) ** 2)
    if mse == 0: return 99.0
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def load_model(cfg, ckpt_path, device):
    vq = cfg['model']['vqgan']
    model = VQGAN3D(
        embedding_dim=vq['embedding_dim'],
        num_embeddings=vq['num_embeddings'],
        base_channels=vq['base_channels'],
        commitment_cost=vq.get('commitment_cost', 0.25),
        disc_weight=vq.get('disc_weight', 0.5),
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.encoder.load_state_dict(ck['encoder'])
    model.decoder.load_state_dict(ck['decoder'])
    model.vq.load_state_dict(ck['vq'])
    if 'discriminator' in ck:
        try: model.discriminator.load_state_dict(ck['discriminator'])
        except Exception: pass
    model.eval()
    print(f"loaded checkpoint epoch {ck.get('epoch','?')}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='/data/yunmin0111/Cor2Vox/configs/c2v_vqgan_pet.yaml')
    ap.add_argument('--ckpt', default='/data/yunmin0111/Cor2Vox/results/c2v/c2v_vqgan_pet/checkpoint/vqgan_last.pth')
    ap.add_argument('--test_dir', default='/data/yunmin0111/dataset_pet_c2v/petTs')
    ap.add_argument('--out', default='/data/yunmin0111/adni_work/pet_recon_test')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    model = load_model(cfg, args.ckpt, device)

    os.makedirs(os.path.join(args.out, 'orig'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'recon'), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.test_dir, '*.nii.gz')))
    if args.limit: files = files[:args.limit]
    print(f"{len(files)} test volumes")

    psnrs = []
    for f in files:
        pid = os.path.basename(f).replace('.nii.gz', '')
        vol = np.asarray(nib.load(f).get_fdata(), dtype=np.float32)  # already [-1,1]
        x = torch.from_numpy(vol).float()[None, None].to(device)     # (1,1,D,H,W)
        with torch.no_grad():
            x_recon, _, _, _ = model(x)
        rec = x_recon[0, 0].cpu().numpy().astype(np.float32)
        rec = np.clip(rec, -1, 1)

        # PSNR over brain voxels (orig > -0.99)
        brain = vol > -0.99
        p = psnr(vol[brain], rec[brain])
        psnrs.append(p)

        nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(args.out, 'orig',  f'{pid}.nii.gz'))
        nib.save(nib.Nifti1Image(rec, np.eye(4)), os.path.join(args.out, 'recon', f'{pid}.nii.gz'))
        print(f"  {pid}: PSNR {p:.2f} dB")

    psnrs = np.array(psnrs)
    print(f"\n=== TEST reconstruction PSNR: {psnrs.mean():.2f} +/- {psnrs.std():.2f} dB "
          f"(n={len(psnrs)}, min {psnrs.min():.2f}, max {psnrs.max():.2f}) ===")


if __name__ == '__main__':
    main()
