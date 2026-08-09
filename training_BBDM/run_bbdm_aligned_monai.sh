#!/usr/bin/bash
#SBATCH -J bbdm_mns
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 4-0
#SBATCH -o /data/%u/logs/bbdm_monai-%A.out

eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v
cd /data/yunmin0111/Cor2Vox
python mri2pet_bbdm_monai.py --epochs 200 --base_ch 64
