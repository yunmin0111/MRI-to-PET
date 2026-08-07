#!/usr/bin/bash
#SBATCH -J vqgan_pet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 4-0
#SBATCH -o /data/%u/logs/vqgan_pet-%A.out

eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v
cd /data/yunmin0111/Cor2Vox
python vqgan3d_v2.py --config configs/c2v_vqgan_pet.yaml --gpu_ids 0
