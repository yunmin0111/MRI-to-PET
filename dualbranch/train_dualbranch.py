# ============================================================================
# train_dualbranch.py
# ----------------------------------------------------------------------------
# 목적:
#   dual-branch BBDM (x_t + y2병변 → PET) 학습.
#   --down 인자로 maxpool(M1)/swin(M2)/zeroconv(M3) 중 하나를 골라 학습한다.
#   나머지(데이터/채널/attention/스케줄/seed 규약)는 세 모델 100% 동일.
#
# 실행 예:
#   python train_dualbranch.py --down maxpool  --seed 0
#   python train_dualbranch.py --down swin     --seed 0
#   python train_dualbranch.py --down zeroconv --seed 0
#
# 데이터:
#   x0 = PET   : dataset_pet_suvr/pet{Tr,Ts}/*.nii.gz   ([-1,1], SUVR 정규화)
#   y1 = MRI   : dataset_pet_suvr/mri{Tr,Ts}/*.nii.gz   ([-1,1])
#   y2 = 병변  : y1 ⊙ mask  (mask = roi_mask_metab, 대사기반 병변 마스크)
#                → 마스크 밖은 -1(배경값)로 채움
#
# 결과:
#   results/dualbranch_{down}_s{seed}/checkpoint/{last.pth, ep*.pth}
#
# 주의(경험적으로 겪은 것들):
#   - GroupNorm(8,c): 채널이 8의 배수여야 함 → base=16이면 OK.
#   - 3D window attention은 D,H,W 가 4의 배수여야 함 → 128 데이터 OK.
#   - GPU 제출 시 --gres=gpu:normal:1 처럼 type 명시가 필요할 수 있음(클러스터 규칙).
#   - 코드 수정 후 반영 안 되면 __pycache__/*.pyc 삭제(캐시가 옛 코드 물고 있음).
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

# 이 파일과 같은 폴더의 모듈 import 가능하게
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from our_dualbranch_bbdm import OurDualBranchBBDM

# ----------------------------------------------------------------------------
# 경로 (환경에 맞게 수정)
# ----------------------------------------------------------------------------
DATA_ROOT = '/data/yunmin0111/dataset_pet_suvr'          # mri{Tr,Ts}, pet{Tr,Ts}
MASK_ROOT = '/data/yunmin0111/adni_work/roi_mask_metab'  # 대사기반 병변 마스크
device = torch.device('cuda')


# ----------------------------------------------------------------------------
# Dataset: (MRI, PET, 병변 y2) 반환
# ----------------------------------------------------------------------------
class VolDS(Dataset):
    def __init__(self, stage):
        sfx = {'train': 'Tr', 'test': 'Ts'}[stage]
        self.md = os.path.join(DATA_ROOT, f'mri{sfx}')
        self.pd = os.path.join(DATA_ROOT, f'pet{sfx}')
        # 마스크는 mriTr/mriTs 하위에 같은 pid 이름으로 있음
        self.mk = os.path.join(MASK_ROOT, f'mri{sfx}')
        self.ids = [os.path.basename(f)[:-7]
                    for f in sorted(glob.glob(f'{self.md}/*.nii.gz'))]
        print(f"[VolDS] {stage}: {len(self.ids)}", flush=True)

    def __len__(self):
        return len(self.ids)

    def _load(self, path):
        return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)

    def __getitem__(self, i):
        pid = self.ids[i]
        mri = self._load(os.path.join(self.md, f'{pid}.nii.gz'))   # [-1,1]
        pet = self._load(os.path.join(self.pd, f'{pid}.nii.gz'))   # [-1,1]
        # 병변 마스크(없을 수 있음 → 전부 배경으로 처리해 안전하게)
        mp = os.path.join(self.mk, f'{pid}.nii.gz')
        if os.path.exists(mp):
            mask = self._load(mp) > 0.5
        else:
            mask = np.zeros_like(mri, dtype=bool)
        # y2 = MRI ⊙ mask, 마스크 밖은 배경값(-1)
        y2 = np.where(mask, mri, -1.0).astype(np.float32)
        # [C=1, D,H,W]
        return (torch.from_numpy(mri)[None],
                torch.from_numpy(pet)[None],
                torch.from_numpy(y2)[None],
                pid)


def psnr(mse, dr=2.0):
    """PSNR. dr=2.0 은 데이터 범위 [-1,1] (폭 2)."""
    return 99.0 if mse <= 0 else 20 * np.log10(dr) - 10 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--down', required=True,
                    choices=['maxpool', 'swin', 'zeroconv'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--no_attn_hi', action='store_true',
                    help='고해상도 window attention 끄기(기본은 켬)')
    a = ap.parse_args()

    # ----- seed 고정 (세 모델이 같은 seed면 동일 초기화/데이터 순서) -----
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    tag = f'dualbranch_{a.down}_s{a.seed}'
    ckpt_dir = f'results/{tag}/checkpoint'
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----- 모델 -----
    model = OurDualBranchBBDM(
        down_kind=a.down, base=a.base,
        use_attn_hi=(not a.no_attn_hi),
        num_timesteps=1000, sample_step=200, max_var=1.0).to(device)
    n_param = sum(p.numel() for p in model.denoise_fn.parameters()) / 1e6
    print(f"[{tag}] UNet params {n_param:.2f}M | attn_hi={not a.no_attn_hi}",
          flush=True)

    opt = torch.optim.AdamW(model.get_parameters(), lr=1e-4, weight_decay=0.0)

    tr = DataLoader(VolDS('train'), batch_size=1, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    te = DataLoader(VolDS('test'), batch_size=1, shuffle=False)

    # ----- 학습 루프 -----
    for ep in range(a.epochs):
        model.denoise_fn.train()
        el = 0.0
        for mri, pet, y2, _ in tr:
            mri, pet, y2 = mri.to(device), pet.to(device), y2.to(device)
            # forward(x0=PET, y1=MRI, y2=병변)
            loss, _ = model(pet, mri, y2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)  # grad clip
            opt.step()
            el += loss.item()
        print(f"Epoch {ep+1}: loss={el/len(tr):.4f}", flush=True)

        # 체크포인트: 매 epoch last, 20마다 백업
        save = {'unet': model.denoise_fn.state_dict(),
                'down': a.down, 'seed': a.seed, 'epoch': ep + 1}
        torch.save(save, f'{ckpt_dir}/last.pth')
        if (ep + 1) % 20 == 0:
            torch.save(save, f'{ckpt_dir}/ep{ep+1}.pth')

        # 10 epoch마다 test 2개로 샘플 PSNR (진척 확인용, 느리니 2개만)
        if (ep + 1) % 10 == 0:
            model.denoise_fn.eval()
            for j, (mri, pet, y2, _) in enumerate(te):
                if j >= 2:
                    break
                mri, pet, y2 = mri.to(device), pet.to(device), y2.to(device)
                with torch.no_grad():
                    gen = model.sample(mri, y2)   # x_T=MRI 에서 PET 생성
                b = (pet > -0.99)                  # 뇌 영역만
                mse = float(((pet[b] - gen[b]) ** 2).mean().item())
                print(f"  [gen]{j}: PSNR={psnr(mse):.1f}dB", flush=True)

    print("DONE", flush=True)


if __name__ == '__main__':
    main()
