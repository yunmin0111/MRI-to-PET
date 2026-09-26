#!/usr/bin/bash
# ============================================================================
# run_all.sh — M1/M2/M3 세 모델을 seed 0으로 각각 SLURM 제출
# ----------------------------------------------------------------------------
# 사용:  bash run_all.sh
# 각 모델당 sbatch 스크립트를 생성해 제출한다. 노드는 -w로 지정(빈 v노드).
# GPU type 명시(normal) 필수 클러스터 규칙 반영: --gres=gpu:normal:1
#
# 주의: 아래 CODE_DIR / 파티션 / account / qos / 노드 를 환경에 맞게 수정.
# ============================================================================

CODE_DIR=/data/yunmin0111/MRI-to-PET-LDM/orig_bbdm   # train_dualbranch.py 위치
PYTHON=/data/yunmin0111/anaconda3/envs/c2v/bin/python3.9
ACCOUNT=ugrad
QOS=qos_yunmin0111_2026_2
PART=batch_ugrad

# 모델별 노드(빈 v노드로 조정). 겹치면 하나씩 대기.
declare -A NODE=( [maxpool]=ariel-v6 [swin]=ariel-v9 [zeroconv]=ariel-v10 )

for DOWN in maxpool swin zeroconv; do
  SH=/data/yunmin0111/run_db_${DOWN}.sh
  cat > $SH << EOF
#!/usr/bin/bash
#SBATCH -J db_${DOWN}
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p ${PART}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH -w ${NODE[$DOWN]}
#SBATCH -t 1-0
#SBATCH -o /data/%u/logs/db_${DOWN}-%A.out
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${CODE_DIR}
${PYTHON} train_dualbranch.py --down ${DOWN} --seed 0 --epochs 200
EOF
  echo "제출: $DOWN (node ${NODE[$DOWN]})"
  sbatch $SH
done

squeue -u $USER
