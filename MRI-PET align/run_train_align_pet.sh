#!/usr/bin/bash
#SBATCH -J mri_align_pet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 4-0
#SBATCH -o /data/%u/logs/mri_align_pet-%A.out

eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v
cd /data/yunmin0111/Cor2Vox
python mri_encoder_align_pet.py --config configs/c2v_mri_align_pet.yaml --gpu_ids 0
