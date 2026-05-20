#!/bin/bash
#SBATCH -J phaseI
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 3:00:00
#SBATCH -o logs/phaseI/phaseI.%j.out
#SBATCH -e logs/phaseI/phaseI.%j.err
#SBATCH --dependency=afterany:21200462

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseI
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
bash scripts/run_phaseI.sh
