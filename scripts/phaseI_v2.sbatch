#!/bin/bash
#SBATCH -J phaseI_v2
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 3:00:00
#SBATCH -o logs/phaseI_v2/phaseI_v2.%j.out
#SBATCH -e logs/phaseI_v2/phaseI_v2.%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseI_v2
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
bash scripts/run_phaseI_v2.sh
