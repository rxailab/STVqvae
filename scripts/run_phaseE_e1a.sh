#!/usr/bin/env bash
# Phase E1a driver: run analyze_wm_teacher_forced.py on every *_best_model.pt
# under the DoorKey directories and emit one JSON per ckpt.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO}/logs/phaseE/teacher_forced}"
N_ROLLOUTS="${N_ROLLOUTS:-200}"
HORIZONS="${HORIZONS:-1 3 5 10}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

mapfile -t CKPTS < <(find \
    "${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0" \
    "${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0" \
    -maxdepth 1 -type f -name '*_best_model.pt' | sort)

echo "Phase E1a: ${#CKPTS[@]} checkpoints"
cd "${REPO}/discrete_mbrl"

for ckpt in "${CKPTS[@]}"; do
  name="$(basename "${ckpt}" .pt)"
  out="${OUT_DIR}/${name}.json"
  log="${OUT_DIR}/${name}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${name}"; continue; fi

  echo "[run]  ${name}"
  set +e
  python analyze_wm_teacher_forced.py \
    --model_path  "${ckpt}" \
    --n_rollouts  "${N_ROLLOUTS}" \
    --horizons    ${HORIZONS} \
    --device      "${DEVICE}" \
    --output_json "${out}" > "${log}" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[fail] ${name} (exit ${rc})"
    rm -f "${out}"
  else
    echo "[ok]   ${name}"
  fi
done
