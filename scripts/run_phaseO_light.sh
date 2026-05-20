#!/usr/bin/env bash
# Phase O-light: dose-response of imagine_ratio on vae_s3.
# Phase O found vae_s3 vanilla=+0.997, oracle-Dyna(ratio=0.5)=+0.544 (-0.45 hurt).
# Test ratios {0.05, 0.1, 0.2, 0.5} × 2 seeds = 8 runs to find the threshold.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseO_light"
DYNA_CKPTS="${CKPT_DIR}/phaseO_light"
mkdir -p "${OUT}" "${DYNA_CKPTS}"
SRC="${CKPT_DIR}/sweep_dk8_vae_s3_best_model.pt"
WM="${CKPT_DIR}/oracle_sweep_dk8_vae_s3_best_model.pt"
cd "${REPO}/discrete_mbrl"

run_dyna() {
  local ratio="$1" seed="$2"
  local out_ckpt="${DYNA_CKPTS}/dyna_r${ratio}_s${seed}.pt"
  local log="${OUT}/dyna_r${ratio}_s${seed}.log"
  if [[ -s "${out_ckpt}" ]]; then echo "[skip] r=${ratio} s=${seed}"; return; fi
  echo "[dyna-O-light] ratio=${ratio} seed=${seed}"
  set +e
  PYTHONHASHSEED=${seed} python train_dyna.py \
    --src_ckpt "${SRC}" --wm_ckpt "${WM}" --wm_mode oracle \
    --output_path "${out_ckpt}" --total_steps 300000 \
    --num_envs 16 --n_steps_per_update 128 \
    --imagine_K 5 --imagine_ratio "${ratio}" \
    --ppo_entropy_coef 0.02 --lr 3e-4 --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] r=${ratio} s=${seed}"; rm -f "${out_ckpt}"; else echo "[ok] r=${ratio} s=${seed}"; fi
}

for ratio in 0.05 0.1 0.2 0.5; do
  for seed in 1 2; do
    run_dyna "${ratio}" "${seed}"
  done
done
