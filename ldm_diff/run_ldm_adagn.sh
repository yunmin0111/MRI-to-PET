#!/usr/bin/bash
#SBATCH -J ldm_ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 4-0
#SBATCH -o /data/%u/logs/ldm_adagn-%A.out
eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v
cd /data/yunmin0111/MRI-to-PET-LDM/ldm_diff
python train_ldm.py --cond adagn --epochs 200
