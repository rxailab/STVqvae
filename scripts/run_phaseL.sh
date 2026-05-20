#!/usr/bin/env bash
# Phase L: bootstrap-only Dyna. The WM is used only as a value-extrapolation
# source — no imagined transitions are added to the PPO buffer. We test
# whether this safer use of the WM beats vanilla PPO (which Phase Dyna showed
# the standard imagined-augmentation does NOT).
#
# Note: Phase Dyna showed PPO collapses by 800k steps with the current hyperparams
# (peak ~+0.85 at 60k steps, decay to ~+0.30 by 800k). Phase L uses 300k steps
# with an entropy floor to keep the comparison in the learning regime, not the
# collapse regime. 4 conditions × 2 seeds = 8 runs.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseL"
DYNA_CKPTS="${CKPT_DIR}/phaseL"
mkdir -p "${OUT}" "${DYNA_CKPTS}"

SRC_CKPT="${CKPT_DIR}/sweep_dk8_vae_s1_best_model.pt"
ORACLE_WM="${CKPT_DIR}/oracle_sweep_dk8_vae_s1_best_model.pt"
CALIB_WM="${CKPT_DIR}/calib_oracle_sweep_dk8_vae_s1_best_model.pt"

cd "${REPO}/discrete_mbrl"

run_dyna() {
  local mode="$1" wm="$2" seed="$3"
  local out_ckpt="${DYNA_CKPTS}/dyna_${mode}_s${seed}.pt"
  local log="${OUT}/dyna_${mode}_s${seed}.log"
  if [[ -s "${out_ckpt}" ]]; then echo "[skip] dyna_${mode}_s${seed}"; return; fi
  echo "[dyna-L] mode=${mode} seed=${seed}"
  set +e
  PYTHONHASHSEED=${seed} python train_dyna.py \
    --src_ckpt "${SRC_CKPT}" \
    --wm_ckpt  "${wm}" \
    --wm_mode  "${mode}" \
    --output_path "${out_ckpt}" \
    --total_steps 300000 \
    --num_envs 16 --n_steps_per_update 128 \
    --imagine_K 5 --imagine_ratio 0.5 \
    --ppo_entropy_coef 0.02 \
    --lr 3e-4 --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] dyna_${mode}_s${seed}"; rm -f "${out_ckpt}"; else echo "[ok] dyna_${mode}_s${seed}"; fi
}

for seed in 1 2; do
  # Vanilla baseline at the new step budget (so all 8 runs are comparable)
  run_dyna none      ""             "${seed}"
  # Standard imagined-augmentation Dyna (oracle WM) — apples-to-apples reference
  run_dyna oracle    "${ORACLE_WM}" "${seed}"
  # Bootstrap-only with oracle WM (Phase L primary test)
  run_dyna bootstrap "${ORACLE_WM}" "${seed}"
  # Bootstrap-only with calibrated WM (do the calibrated heads matter when
  # the WM is only used for bootstrapping?)
  run_dyna bootstrap "${CALIB_WM}"  "${seed}"
done
