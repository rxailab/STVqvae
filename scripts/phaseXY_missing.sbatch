#!/bin/bash
#SBATCH -J phaseXY_miss
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 1:30:00
#SBATCH -o logs/phaseXY/phaseXY_miss.%j.out
#SBATCH -e logs/phaseXY/phaseXY_miss.%j.err
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
# Just rerun the script — the [skip] guards will reuse the 7 completed cells
# and only finish the missing crafter lam=10.0 oracle/Phase A/Phase N stages.
bash scripts/run_phaseXY.sh
