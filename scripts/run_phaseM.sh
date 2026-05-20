#!/usr/bin/env bash
# Phase M — action-Q ranking under the WM. Tests whether the WM can rank
# ACTIONS at a state (Dreamer/MuZero use case) even though it can't rank
# returns across episodes (per Phase K).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseM"
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
  "conv_vae_s1_h64          ${CKPT_DIR}/conv_oracle_sweep_dk8_vae_s1_best_model_h64.pt"
  "conv_vae_s2_h64          ${CKPT_DIR}/conv_oracle_sweep_dk8_vae_s2_best_model_h64.pt"
  "conv_vae_s3_h64          ${CKPT_DIR}/conv_oracle_sweep_dk8_vae_s3_best_model_h64.pt"
)
cd "${REPO}/discrete_mbrl"
for entry in "${WMS[@]}"; do
  label="${entry%% *}"; ckpt="${entry##* }"
  out="${OUT}/phaseM_${label}.json"; log="${OUT}/phaseM_${label}.log"
  if [[ ! -f "${ckpt}" ]]; then echo "[skip-miss] ${label}"; continue; fi
  if [[ -s "${out}" ]]; then echo "[skip] ${label}"; continue; fi
  echo "[run] ${label}"
  set +e
  python analyze_wm_action_q.py --model_path "${ckpt}" \
    --n_states 4000 --n_eval_eps 100 --max_steps 200 \
    --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${label}"; rm -f "${out}"; else echo "[ok] ${label}"; fi
done
