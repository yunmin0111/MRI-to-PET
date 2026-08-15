#!/usr/bin/bash
#SBATCH -J vq_m2p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 4-0
#SBATCH -o /data/%u/logs/vqgan_mri2pet-%A.out

eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v
cd /data/yunmin0111/Cor2Vox
python vqgan3d_mri2pet.py --config configs/c2v_vqgan_mri2pet.yaml --gpu_ids 0
