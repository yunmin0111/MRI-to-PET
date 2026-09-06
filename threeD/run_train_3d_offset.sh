#!/usr/bin/bash
#SBATCH -J ae3d_off
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 4-0
#SBATCH -o /data/%u/logs/ae3d_offset-%A.out
eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v
cd /data/yunmin0111/MRI-to-PET-LDM/LDM_AE_offset/threeD
python train_ae_3d.py --config ae_3d.yaml --attn offset --gpu 0
