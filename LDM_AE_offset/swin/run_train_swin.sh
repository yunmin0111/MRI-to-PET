#!/usr/bin/bash
#SBATCH -J ae_swin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 4-0
#SBATCH -o /data/%u/logs/ae_swin-%A.out

eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v
cd /data/yunmin0111/MRI-to-PET-LDM/LDM_AE_offset
python train_ae_swin.py --config ae_swin_2d.yaml --gpu 0
