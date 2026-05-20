#!/bin/bash
#SBATCH -J phII_cC
#SBATCH -p gpu-long
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o logs/phaseII_cellC/phaseII_cellC.%j.out
#SBATCH -e logs/phaseII_cellC/phaseII_cellC.%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseII_cellC
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${REPO}/.mplconfig"
export MPLCONFIGDIR="${REPO}/.mplconfig"
bash scripts/run_phaseII_cellC.sh
