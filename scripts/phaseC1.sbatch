#!/bin/bash
#SBATCH -J phaseC1
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o logs/phaseC1/phaseC1.%A_%a.out
#SBATCH -e logs/phaseC1/phaseC1.%A_%a.err
#SBATCH --array=1-3

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseC1

bash scripts/phaseC1_train_one.sh "${SLURM_ARRAY_TASK_ID}"
