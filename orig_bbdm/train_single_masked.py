# ============================================================================
# train_single_masked.py
# ----------------------------------------------------------------------------
# 목적 (비교용 baseline):
#   pixel space BBDM, bridge 종점 y = MRI ⊙ mask (병변 영역 MRI만) → PET 생성.
#   "병변 영역 MRI만 주면 PET이 얼마나 나오나"를 재기 위한 모델.
#   dual-branch가 아니다: 네트워크 입력은 x_t 하나.
#
# BBDM 수식 (dual-branch 래퍼와 완전히 동일, objective='grad'):
#   y      = MRI ⊙ mask   (마스크 밖 = -1, 배경값)
#   x_t    = (1−m)·x0 + m·y + σ·ε
#   target = m·(y − x0) + σ·ε          (= x_t − x0)
#   loss   = L1( f_θ(x_t, t), target )
#   샘플링: x_T = y(=MRI⊙mask) 에서 출발 → x_0 = 생성 PET
#
#   ※ dual-branch에선 y=전체 MRI이고 MRI⊙mask는 별도 조건(B branch)이었다.
#     여기선 전체 MRI 정보가 아예 없다. 즉 마스크 밖 영역의 PET은
#     입력 정보 없이 생성해야 하므로, 뇌 전체 지표보다 ROI 지표가 핵심 비교 대상.
#
# 공정 비교를 위해 dual-branch와 맞춘 것:
#   - 데이터: dataset_pet_suvr (MNI + SUVR)
#   - 마스크: roi_mask_metab (대사 기반 병변 마스크) — dual의 y2와 동일
#   - UNet 블록/해상도/attention/down 모듈 — singlebranch_unet.py 참고
#   - BBDM 스케줄: T=1000, m_t=t/T(양끝 clamp), var_t=2s(m−m²), s=1.0, 200-step 샘플링
#   - 옵티마이저: AdamW lr 1e-4, grad clip 1.0, seed
#
# 실행:
#   python train_single_masked.py --down maxpool --seed 0
# 결과:
#   results/single_masked_{down}_s{seed}/checkpoint/{last,ep*}.pth
# ============================================================================

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from singlebranch_unet import SingleBranchUNet

DATA_ROOT = '/data/yunmin0111/dataset_pet_suvr'
MASK_ROOT = '/data/yunmin0111/adni_work/roi_mask_metab'
device = torch.device('cuda')


# ----------------------------------------------------------------------------
# BBDM 래퍼 (single-branch). 스케줄/샘플링 식은 our_dualbranch_bbdm.py와 동일.
# ----------------------------------------------------------------------------
class SingleMaskedBBDM(nn.Module):
    def __init__(self, down_kind='maxpool', base=16, use_attn_hi=True,
                 num_timesteps=1000, sample_step=200, max_var=1.0):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.sample_step = sample_step
        self.eta = 1.0
        self.denoise_fn = SingleBranchUNet(down_kind, base, use_attn_hi)

        T = num_timesteps
        m_t = np.linspace(0.0, 1.0, T + 1, dtype=np.float64)
        m_t[0] = 1e-4; m_t[-1] = 1.0 - 1e-4          # 양끝 0/1 회피 (dual과 동일)
        var_t = 2.0 * max_var * (m_t - m_t ** 2)
        self.register_buffer('m_t', torch.tensor(m_t, dtype=torch.float32))
        self.register_buffer('variance_t', torch.tensor(var_t, dtype=torch.float32))

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x0, y):
        """x0 = PET, y = MRI⊙mask"""
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        m = self.m_t[t].view(b, 1, 1, 1, 1)
        sigma = torch.sqrt(self.variance_t[t].clamp(min=0)).view(b, 1, 1, 1, 1)
        x_t = (1. - m) * x0 + m * y + sigma * noise          # 같은 noise 사용
        target = m * (y - x0) + sigma * noise                # = x_t - x0
        pred = self.denoise_fn(x_t, timesteps=t)
        return (target - pred).abs().mean()

    @torch.no_grad()
    def sample(self, y):
        """x_T = y(MRI⊙mask) 에서 역과정. oracle 검증된 계수와 동일 구조."""
        b, dev = y.shape[0], y.device
        steps = list(reversed(np.linspace(0, self.num_timesteps,
                                          self.sample_step + 1, dtype=int).tolist()))
        x_t = y.clone()
        for i in range(len(steps) - 1):
            cs, ns = steps[i], steps[i + 1]
            t = torch.full((b,), cs, device=dev, dtype=torch.long)
            x0r = x_t - self.denoise_fn(x_t, timesteps=t)     # x0 추정
            if ns == 0:
                x_t = x0r
            else:
                mt, mnt = self.m_t[cs], self.m_t[ns]
                vt, vnt = self.variance_t[cs], self.variance_t[ns]
                s2 = torch.clamp((vt - vnt * (1. - mt) ** 2 / ((1. - mnt) ** 2 + 1e-8))
                                 * vnt / (vt + 1e-8), min=0)
                x_t = ((1. - mnt) * x0r + mnt * y
                       + torch.sqrt(torch.clamp((vnt - s2) / (vt + 1e-8), min=0))
                       * (x_t - (1. - mt) * x0r - mt * y)
                       + torch.sqrt(s2) * self.eta * torch.randn_like(x_t))
        return x_t


# ----------------------------------------------------------------------------
# Dataset: (y = MRI⊙mask, PET, mask, pid)
# ----------------------------------------------------------------------------
class MaskedDS(Dataset):
    def __init__(self, stage):
        sfx = {'train': 'Tr', 'test': 'Ts'}[stage]
        self.md = os.path.join(DATA_ROOT, f'mri{sfx}')
        self.pd = os.path.join(DATA_ROOT, f'pet{sfx}')
        self.mk = os.path.join(MASK_ROOT, f'mri{sfx}')
        # 마스크가 있는 subject만 사용 (마스크 없는 경우 y가 전부 -1이 되므로 제외)
        self.ids = [os.path.basename(f)[:-7]
                    for f in sorted(glob.glob(f'{self.md}/*.nii.gz'))
                    if os.path.exists(os.path.join(self.mk, os.path.basename(f)))]
        print(f"[MaskedDS] {stage}: {len(self.ids)}", flush=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        ld = lambda p: np.asarray(nib.load(p).get_fdata(), dtype=np.float32)
        mri = ld(os.path.join(self.md, f'{pid}.nii.gz'))
        pet = ld(os.path.join(self.pd, f'{pid}.nii.gz'))
        mask = ld(os.path.join(self.mk, f'{pid}.nii.gz')) > 0.5
        y = np.where(mask, mri, -1.0).astype(np.float32)      # MRI⊙mask, 밖은 -1
        return (torch.from_numpy(y)[None], torch.from_numpy(pet)[None],
                torch.from_numpy(mask.astype(np.float32))[None], pid)


def psnr(mse, dr=2.0):
    return 99.0 if mse <= 0 else 20 * np.log10(dr) - 10 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--down', default='maxpool', choices=['maxpool', 'swin', 'zeroconv'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--no_attn_hi', action='store_true')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    tag = f'single_masked_{a.down}_s{a.seed}'
    ckpt_dir = f'results/{tag}/checkpoint'
    os.makedirs(ckpt_dir, exist_ok=True)

    model = SingleMaskedBBDM(down_kind=a.down, use_attn_hi=not a.no_attn_hi).to(device)
    print(f"[{tag}] UNet params "
          f"{sum(p.numel() for p in model.denoise_fn.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.get_parameters(), lr=1e-4, weight_decay=0.0)
    tr = DataLoader(MaskedDS('train'), batch_size=1, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    te = DataLoader(MaskedDS('test'), batch_size=1, shuffle=False)

    for ep in range(a.epochs):
        model.denoise_fn.train()
        el = 0.0
        for y, pet, _, _ in tr:
            y, pet = y.to(device), pet.to(device)
            loss = model(pet, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
            opt.step()
            el += loss.item()
        print(f"Epoch {ep+1}: loss={el/len(tr):.4f}", flush=True)

        ck = {'unet': model.denoise_fn.state_dict(), 'down': a.down,
              'seed': a.seed, 'epoch': ep + 1}
        torch.save(ck, f'{ckpt_dir}/last.pth')
        if (ep + 1) % 20 == 0:
            torch.save(ck, f'{ckpt_dir}/ep{ep+1}.pth')

        # 10 epoch마다 test 2개: 뇌 전체 PSNR + ROI(마스크 내부) PSNR
        if (ep + 1) % 10 == 0:
            model.denoise_fn.eval()
            for j, (y, pet, mask, _) in enumerate(te):
                if j >= 2:
                    break
                y, pet, mask = y.to(device), pet.to(device), mask.to(device) > 0.5
                gen = model.sample(y)
                b = pet > -0.99
                print(f"  [gen]{j}: PSNR={psnr(float(((pet[b]-gen[b])**2).mean())):.1f}dB "
                      f"ROI_PSNR={psnr(float(((pet[mask]-gen[mask])**2).mean())):.1f}dB",
                      flush=True)
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
