#!/usr/bin/env bash
# Phase S: action-conditioning probe on DK-8x8 VQ-VAE encoders.
# Tests whether the encoder-determined action-blindness from Phase N
# (continuous spatial VAE) reproduces on DK-8x8 with discrete VQ encoders.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseS"
mkdir -p "${OUT}"
WMS=(
  "v5dc_s4              ${CKPT_DIR}/sweep_dk8_v5dc_s4_best_model.pt"
  "v5dc_s5              ${CKPT_DIR}/sweep_dk8_v5dc_s5_best_model.pt"
  "v6_s4                ${CKPT_DIR}/sweep_dk8_v6_s4_best_model.pt"
  "v6_s5                ${CKPT_DIR}/sweep_dk8_v6_s5_best_model.pt"
  "oracle_v5dc_s4       ${CKPT_DIR}/oracle_sweep_dk8_v5dc_s4_best_model.pt"
  "oracle_v6_s4         ${CKPT_DIR}/calib_oracle_sweep_dk8_v6_s4_best_model.pt"
  "oracle_v6_s5         ${CKPT_DIR}/calib_oracle_sweep_dk8_v6_s5_best_model.pt"
  "oracle_rigor_v6_s1   ${CKPT_DIR}/oracle_rigor_dk8_v6_s1_best_model.pt"
  "oracle_rigor_v6_s2   ${CKPT_DIR}/oracle_rigor_dk8_v6_s2_best_model.pt"
)
cd "${REPO}/discrete_mbrl"
for entry in "${WMS[@]}"; do
  label="${entry%% *}"; ckpt="${entry##* }"
  out="${OUT}/phaseS_${label}.json"; log="${OUT}/phaseS_${label}.log"
  if [[ ! -f "${ckpt}" ]]; then echo "[skip-miss] ${label}"; continue; fi
  if [[ -s "${out}" ]]; then echo "[skip] ${label}"; continue; fi
  echo "[run] ${label}"
  set +e
  python analyze_wm_action_cond.py --model_path "${ckpt}" \
    --n_states 2000 --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${label}"; rm -f "${out}"; else echo "[ok] ${label}"; fi
done
