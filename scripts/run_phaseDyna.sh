#!/usr/bin/env bash
# Full Dyna sweep: train fresh policies via PPO with WM-augmented imagined
# rollouts. 4 conditions x 2 seeds = 8 runs.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseDyna"
DYNA_CKPTS="${CKPT_DIR}/dyna"
mkdir -p "${OUT}" "${DYNA_CKPTS}"

# Use vae_s1 as the canonical source encoder (fully working policy + WM cohort)
SRC_CKPT="${CKPT_DIR}/sweep_dk8_vae_s1_best_model.pt"
ONLINE_WM="${SRC_CKPT}"
ORACLE_WM="${CKPT_DIR}/oracle_sweep_dk8_vae_s1_best_model.pt"
CALIB_WM="${CKPT_DIR}/calib_oracle_sweep_dk8_vae_s1_best_model.pt"

cd "${REPO}/discrete_mbrl"

run_dyna() {
  local mode="$1" wm="$2" seed="$3"
  local out_ckpt="${DYNA_CKPTS}/dyna_${mode}_s${seed}.pt"
  local log="${OUT}/dyna_${mode}_s${seed}.log"
  if [[ -s "${out_ckpt}" ]]; then echo "[skip] dyna_${mode}_s${seed}"; return; fi
  echo "[dyna] mode=${mode} seed=${seed}"
  set +e
  PYTHONHASHSEED=${seed} python train_dyna.py \
    --src_ckpt "${SRC_CKPT}" \
    --wm_ckpt  "${wm}" \
    --wm_mode  "${mode}" \
    --output_path "${out_ckpt}" \
    --total_steps 800000 \
    --num_envs 16 --n_steps_per_update 128 \
    --imagine_K 5 --imagine_ratio 0.5 \
    --lr 3e-4 --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] dyna_${mode}_s${seed}"; rm -f "${out_ckpt}"; else echo "[ok] dyna_${mode}_s${seed}"; fi
}

for seed in 1 2; do
  # vanilla PPO baseline (no WM augmentation)
  run_dyna none "" "${seed}"
  # online-trained WM (paper baseline)
  run_dyna online "${ONLINE_WM}" "${seed}"
  # offline-trained oracle WM (Phase E2)
  run_dyna oracle "${ORACLE_WM}" "${seed}"
  # head-calibrated oracle WM (Phase I)
  run_dyna calib "${CALIB_WM}" "${seed}"
done
