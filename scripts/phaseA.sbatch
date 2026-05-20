#!/bin/bash
#SBATCH -J phaseA_e1
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/phaseA/%x.%j.out
#SBATCH -e logs/phaseA/%x.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseA

# Activate the env that has torch + sklearn + minigrid
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8

# Default scans the two DoorKey directories. Pass an explicit list as $1 to
# narrow scope, e.g. `sbatch scripts/phaseA.sbatch <dir1>,<dir2>`.
bash scripts/run_phaseA.sh "${1:-}" "${REPO}/logs/phaseA"
