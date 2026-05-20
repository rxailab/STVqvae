#!/bin/bash
#SBATCH -J phaseE_full
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/phaseE/phaseE_full.%j.out
#SBATCH -e logs/phaseE/phaseE_full.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseE

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8

echo "=== Phase E1a: teacher-forced sweep ==="
date
bash scripts/run_phaseE_e1a.sh "${REPO}/logs/phaseE/teacher_forced"

echo
echo "=== Phase E2: oracle WM sweep (5 ckpts) ==="
date
bash scripts/run_phaseE_e2.sh "${REPO}/logs/phaseE/oracle_analyzer"

echo
echo "DONE"
date
