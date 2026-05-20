#!/bin/bash
#SBATCH -J crafter_sweep
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o logs/crafter/crafter_sweep.%j.out
#SBATCH -e logs/crafter/crafter_sweep.%j.err
#SBATCH --dependency=afterany:21216606

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/crafter
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
bash scripts/run_crafter_sweep.sh
