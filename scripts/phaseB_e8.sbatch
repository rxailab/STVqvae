#!/bin/bash
#SBATCH -J phaseB_e8
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 2:00:00
#SBATCH -o logs/phaseB/phaseB_e8.%j.out
#SBATCH -e logs/phaseB/phaseB_e8.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseB/e8

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8

bash scripts/run_phaseB_e8.sh "" "${REPO}/logs/phaseB/e8"
