#!/usr/bin/bash
#SBATCH -J synthstrip_pet
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 1-00:00:00
#SBATCH -o /data/yunmin0111/logs/synthstrip_pet-%A.out

eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v

MODEL=/data/yunmin0111/synthstrip/synthstrip.1.pt
SCRIPT=/data/yunmin0111/synthstrip/mri_synthstrip
PETDIR=/data/yunmin0111/adni_work/pet_nii
OUT=/data/yunmin0111/adni_work/pet_brain
mkdir -p "$OUT"

python3 - << 'PY'
import pandas as pd, os, subprocess
df = pd.read_csv('/data/yunmin0111/adni_work/adni_pairs_manifest.csv')
model = '/data/yunmin0111/synthstrip/synthstrip.1.pt'
script = '/data/yunmin0111/synthstrip/mri_synthstrip'
out = '/data/yunmin0111/adni_work/pet_brain'
n_ok, n_skip = 0, 0
for i, r in df.iterrows():
    pid = str(r['PET_ImageID'])
    src = r['pet_path']                     # /.../pet_nii/I240519.nii.gz
    mask = os.path.join(out, f'{pid}_mask.nii.gz')
    if os.path.exists(mask):
        n_skip += 1; continue
    cmd = ['python3', script, '-i', src,
           '-o', os.path.join(out, f'{pid}.nii.gz'),
           '-m', mask, '--model', model, '-b', '2']
    r_ = subprocess.run(cmd, capture_output=True, text=True)
    if r_.returncode == 0 and os.path.exists(mask):
        n_ok += 1
    else:
        print(f'FAILED {pid}: {r_.stderr[-200:]}', flush=True)
    if (i+1) % 25 == 0:
        print(f'[{i+1}/{len(df)}] ok={n_ok} skip={n_skip}', flush=True)
print(f'DONE ok={n_ok} skip={n_skip}')
PY

echo "=== PET mask files ==="; ls "$OUT"/*_mask.nii.gz 2>/dev/null | wc -l
echo "===== Done ====="
