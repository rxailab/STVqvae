#!/bin/bash
#SBATCH -J phaseZ
#SBATCH -p gpu-long
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/phaseZ/phaseZ.%j.out
#SBATCH -e logs/phaseZ/phaseZ.%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
bash scripts/run_phaseZ.sh
