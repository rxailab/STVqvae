#!/usr/bin/env bash
# Phase U: action-conditioning probe on DK-16 ckpts. Tests whether the
# four-regime taxonomy from Phase O on DK-8 holds on the harder DK-16 task.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
OUT="${REPO}/logs/phaseU"
mkdir -p "${OUT}"
WMS=(
  "v6_s1                ${CKPT_DIR}/sweep_doorkey16_v6_s1_best_model.pt"
  "v6_s2                ${CKPT_DIR}/sweep_doorkey16_v6_s2_best_model.pt"
  "v6_s3                ${CKPT_DIR}/sweep_doorkey16_v6_s3_best_model.pt"
  "v5dc_s1              ${CKPT_DIR}/sweep_doorkey16_v5dc_s1_best_model.pt"
  "v5dc_s2              ${CKPT_DIR}/sweep_doorkey16_v5dc_s2_best_model.pt"
  "v5dc_s3              ${CKPT_DIR}/sweep_doorkey16_v5dc_s3_best_model.pt"
  "v6_cb1024_s1         ${CKPT_DIR}/dk16_v6_cb1024_s1_best_model.pt"
  "rigor_cb512          ${CKPT_DIR}/rigor_dk16_cb512_thr2_best_model.pt"
)
cd "${REPO}/discrete_mbrl"
for entry in "${WMS[@]}"; do
  label="${entry%% *}"; ckpt="${entry##* }"
  out="${OUT}/phaseU_${label}.json"; log="${OUT}/phaseU_${label}.log"
  if [[ ! -f "${ckpt}" ]]; then echo "[skip-miss] ${label}"; continue; fi
  if [[ -s "${out}" ]]; then echo "[skip] ${label}"; continue; fi
  echo "[run] ${label}"
  set +e
  python analyze_wm_action_cond.py --model_path "${ckpt}" \
    --n_states 2000 --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${label}"; rm -f "${out}"; else echo "[ok] ${label}"; fi
done
