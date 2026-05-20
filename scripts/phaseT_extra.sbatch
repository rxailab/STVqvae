#!/bin/bash
#SBATCH -J phaseT_extra
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/phaseT/phaseT_extra.%j.out
#SBATCH -e logs/phaseT/phaseT_extra.%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${REPO}/.mplconfig" "${REPO}/logs/phaseT"
export MPLCONFIGDIR="${REPO}/.mplconfig"
bash scripts/run_phaseT_extra.sh
