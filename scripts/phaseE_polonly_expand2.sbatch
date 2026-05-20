#!/bin/bash
#SBATCH -J phaseE_extra
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -o logs/phaseE/phaseE_extra.%j.out
#SBATCH -e logs/phaseE/phaseE_extra.%j.err
#SBATCH --dependency=afterany:21216486

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseE
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8

echo "=== Action 1: policy-only-buffer control ==="
bash scripts/run_phaseE_polonly.sh

echo
echo "=== Action 2: N=15 expansion (5 more mixed-buffer ckpts) ==="
bash scripts/run_phaseE_expand2.sh
