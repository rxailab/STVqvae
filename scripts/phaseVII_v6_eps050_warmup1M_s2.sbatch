#!/bin/bash
#SBATCH -J phVII_w2
#SBATCH -p gpu-long
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/phaseVII/phaseVII_v6_eps050_warmup1M_s2.%j.out
#SBATCH -e logs/phaseVII/phaseVII_v6_eps050_warmup1M_s2.%j.err

set -euo pipefail
REPO="/mmfs1/storage/users/xiar3/exp/STVqvae"
cd "${REPO}"
mkdir -p logs/phaseVII
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${REPO}/.mplconfig"
export MPLCONFIGDIR="${REPO}/.mplconfig"
bash scripts/run_phaseVII_v6.sh v6_eps050_warmup1M_s2 2 0.50 '--encoder_lr 1e-5 --encoder_lr_cosine --explore_warmup_steps 1000000'
