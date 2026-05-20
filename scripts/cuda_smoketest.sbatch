#!/bin/bash
#SBATCH -J cuda_check
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:10:00
#SBATCH -o logs/phaseA/cuda_check.%j.out
#SBATCH -e logs/phaseA/cuda_check.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseA

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5

echo "=== nvidia-smi ==="
nvidia-smi || true
echo
echo "=== python / torch ==="
python - <<'PY'
import torch, sys
print('python   :', sys.version.split()[0])
print('torch    :', torch.__version__)
print('cuda built:', torch.version.cuda)
print('cuda avail:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device   :', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    x = torch.zeros(1024, 1024, device='cuda')
    y = (x + 1).matmul(x + 1)
    torch.cuda.synchronize()
    print('matmul ok, sum =', float(y.sum()))
else:
    print('NO CUDA -- aborting smoke test')
    sys.exit(1)
PY
echo
echo "DONE"
