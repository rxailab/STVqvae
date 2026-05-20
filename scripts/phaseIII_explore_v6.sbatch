#!/bin/bash
#SBATCH -J phIII_v6
#SBATCH -p gpu-long
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 5:00:00
#SBATCH -o logs/phaseIII/phaseIII_v6.%j.out
#SBATCH -e logs/phaseIII/phaseIII_v6.%j.err

set -euo pipefail
REPO="/mmfs1/storage/users/xiar3/exp/STVqvae"
cd "${REPO}"
mkdir -p logs/phaseIII
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${REPO}/.mplconfig"
export MPLCONFIGDIR="${REPO}/.mplconfig"
bash scripts/run_phaseIII_explore_v6.sh
