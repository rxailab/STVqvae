#!/bin/bash
#SBATCH -J phaseE_arch
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o logs/phaseE/arch/phaseE_arch.%j.out
#SBATCH -e logs/phaseE/arch/phaseE_arch.%j.err
#SBATCH --dependency=afterany:21216638

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseE/arch
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
bash scripts/run_phaseE_arch.sh
