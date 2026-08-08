#!/usr/bin/env python3
"""
Reconstruct held-out TEST PET volumes through the trained PET VQGAN.
Metrics: PSNR, SSIM, MAE, MSE (computed over brain voxels).
Saves original + reconstruction and a per-volume metrics CSV.
"""
import os, sys, glob, yaml, argparse, csv
import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, '/data/yunmin0111/Cor2Vox')
from vqgan3d import VQGAN3D

try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


def psnr(a, b, data_range=2.0):
    mse = np.mean((a - b) ** 2)
    if mse == 0: return 99.0
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim_vol(orig, rec, mask, data_range=2.0):
    """3D SSIM. skimage if available (whole volume), else masked fallback."""
    if HAVE_SKIMAGE:
        # SSIM needs full volume; compute on whole then it's fine (bg is -1 in both)
        val = sk_ssim(orig, rec, data_range=data_range)
        return float(val)
    # fallback: simple global SSIM on brain voxels
    a, b = orig[mask], rec[mask]
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * data_range) ** 2; c2 = (0.03 * data_range) ** 2
    return float(((2*mu_a*mu_b + c1)*(2*cov + c2)) / ((mu_a**2 + mu_b**2 + c1)*(va + vb + c2)))


def load_model(cfg, ckpt_path, device):
    vq = cfg['model']['vqgan']
    model = VQGAN3D(
        in_channels=1,
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
    print(f"loaded checkpoint epoch {ck.get('epoch','?')} | skimage SSIM: {HAVE_SKIMAGE}")
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
    print(f"{len(files)} test volumes\n")

    rows = []
    for f in files:
        pid = os.path.basename(f).replace('.nii.gz', '')
        vol = np.asarray(nib.load(f).get_fdata(), dtype=np.float32)
        x = torch.from_numpy(vol).float()[None, None].to(device)
        with torch.no_grad():
            x_recon, _, _, _ = model(x)
        rec = np.clip(x_recon[0, 0].cpu().numpy().astype(np.float32), -1, 1)

        brain = vol > -0.99
        o, r = vol[brain], rec[brain]
        mse = float(np.mean((o - r) ** 2))
        mae = float(np.mean(np.abs(o - r)))
        ps  = psnr(o, r)
        ss  = ssim_vol(vol, rec, brain)
        rows.append({'id': pid, 'PSNR': ps, 'SSIM': ss, 'MAE': mae, 'MSE': mse})

        nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(args.out, 'orig',  f'{pid}.nii.gz'))
        nib.save(nib.Nifti1Image(rec, np.eye(4)), os.path.join(args.out, 'recon', f'{pid}.nii.gz'))
        print(f"  {pid}: PSNR {ps:.2f} | SSIM {ss:.4f} | MAE {mae:.4f} | MSE {mse:.5f}")

    # CSV
    csv_path = os.path.join(args.out, 'metrics.csv')
    with open(csv_path, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=['id', 'PSNR', 'SSIM', 'MAE', 'MSE'])
        w.writeheader(); w.writerows(rows)

    def stat(k):
        v = np.array([r[k] for r in rows])
        return v.mean(), v.std(), v.min(), v.max()
    print("\n=== TEST reconstruction metrics (n=%d, brain voxels) ===" % len(rows))
    for k, unit in [('PSNR', 'dB'), ('SSIM', ''), ('MAE', ''), ('MSE', '')]:
        m, s, lo, hi = stat(k)
        print(f"  {k:5s}: {m:.4f} +/- {s:.4f} {unit}  (min {lo:.4f}, max {hi:.4f})")
    print(f"\nwrote {csv_path}")


if __name__ == '__main__':
    main()
