#!/usr/bin/env bash
# Phase K' — rerun analyze_wm_planning with the V_critic baseline patch.
# Same WMs as Phase K. Outputs go to logs/phaseKp/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseKp"
mkdir -p "${OUT}"
WMS=(
  "online_vae_s1            ${CKPT_DIR}/sweep_dk8_vae_s1_best_model.pt"
  "oracle_vae_s1            ${CKPT_DIR}/oracle_sweep_dk8_vae_s1_best_model.pt"
  "oracle_vae_s2            ${CKPT_DIR}/oracle_sweep_dk8_vae_s2_best_model.pt"
  "oracle_vae_s3            ${CKPT_DIR}/oracle_sweep_dk8_vae_s3_best_model.pt"
  "calib_vae_s1             ${CKPT_DIR}/calib_oracle_sweep_dk8_vae_s1_best_model.pt"
  "calib_vae_s2             ${CKPT_DIR}/calib_oracle_sweep_dk8_vae_s2_best_model.pt"
  "multistep_vae_s1         ${CKPT_DIR}/multistep_sweep_dk8_vae_s1_best_model.pt"
  "multistep_vae_s2         ${CKPT_DIR}/multistep_sweep_dk8_vae_s2_best_model.pt"
  "multistep_vae_s3         ${CKPT_DIR}/multistep_sweep_dk8_vae_s3_best_model.pt"
  "conv_vae_s1_h32          ${CKPT_DIR}/conv_oracle_sweep_dk8_vae_s1_best_model_h32.pt"
  "conv_vae_s1_h64          ${CKPT_DIR}/conv_oracle_sweep_dk8_vae_s1_best_model_h64.pt"
  "conv_vae_s2_h64          ${CKPT_DIR}/conv_oracle_sweep_dk8_vae_s2_best_model_h64.pt"
  "conv_vae_s3_h64          ${CKPT_DIR}/conv_oracle_sweep_dk8_vae_s3_best_model_h64.pt"
)
cd "${REPO}/discrete_mbrl"
for entry in "${WMS[@]}"; do
  label="${entry%% *}"; ckpt="${entry##* }"
  out="${OUT}/phaseKp_${label}.json"; log="${OUT}/phaseKp_${label}.log"
  if [[ ! -f "${ckpt}" ]]; then echo "[skip-miss] ${label}"; continue; fi
  if [[ -s "${out}" ]]; then echo "[skip] ${label}"; continue; fi
  echo "[run] ${label}"
  set +e
  python analyze_wm_planning.py --model_path "${ckpt}" \
    --n_episodes 300 --max_steps 200 --horizons 1 3 5 10 \
    --stochastic --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${label}"; rm -f "${out}"; else echo "[ok] ${label}"; fi
done
