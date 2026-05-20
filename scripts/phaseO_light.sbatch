#!/bin/bash
#SBATCH -J phaseO_light
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/phaseO_light/phaseO_light.%j.out
#SBATCH -e logs/phaseO_light/phaseO_light.%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseO_light
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
bash scripts/run_phaseO_light.sh
