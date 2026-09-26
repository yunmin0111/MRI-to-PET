#!/usr/bin/bash
# ============================================================================
# run_single_masked.sh — pixel BBDM, 입력 y = MRI⊙mask (single-branch, 비교용)
# down 모듈은 dual-branch 1차 실행과 같은 maxpool, seed 0.
# 노드(-w)는 빈 v노드로 바꿔서 제출.
# ============================================================================
#SBATCH -J single_mask
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p batch_ugrad
#SBATCH --account=ugrad
#SBATCH --qos=qos_yunmin0111_2026_2
#SBATCH -w ariel-v9
#SBATCH -t 3-0
#SBATCH -o /data/%u/logs/single_mask-%A.out
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yunmin0111/MRI-to-PET-LDM/orig_bbdm
/data/yunmin0111/anaconda3/envs/c2v/bin/python3.9 train_single_masked.py --down maxpool --seed 0 --epochs 200
