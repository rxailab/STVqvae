#!/usr/bin/env bash
# Phase D-light driver: run analyze_wm_planning.py on a list of ckpts.
# Default scope: DK-8 best_model ckpts likely to have a working policy.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO}/logs/phaseD/sweep}"
N_EPISODES="${N_EPISODES:-100}"
MAX_STEPS="${MAX_STEPS:-150}"
HORIZONS="${HORIZONS:-5 10 20}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

# Subset: working-policy DK-8 ckpts. Skip DK-16 (policies don't converge) and
# skip rigor_dk8_vae (no transition model).
mapfile -t CKPTS < <(find \
    "${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0" \
    -maxdepth 1 -type f -name '*_best_model.pt' \
    | grep -E '(sweep_dk8_v6|sweep_dk8_v2|sweep_dk8_v5dc|sweep_dk8_vae|sweep_doorkey8)' \
    | grep -v 'oracle_' \
    | grep -v 'phaseC1_' \
    | sort)

cd "${REPO}/discrete_mbrl"
echo "Phase D-light: ${#CKPTS[@]} checkpoints"

for ckpt in "${CKPTS[@]}"; do
  name="$(basename "${ckpt}" .pt)"
  out="${OUT_DIR}/${name}.json"
  log="${OUT_DIR}/${name}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${name}"; continue; fi

  echo "[run]  ${name}"
  set +e
  python analyze_wm_planning.py \
    --model_path  "${ckpt}" \
    --n_episodes  "${N_EPISODES}" \
    --max_steps   "${MAX_STEPS}" \
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
