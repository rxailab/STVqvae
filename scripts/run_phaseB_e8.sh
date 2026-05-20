#!/usr/bin/env bash
# Phase B E8 driver: run analyze_wm_phaseB_e8.py on a list (or directory) of ckpts.
# Default scope: DK-16 v6 (anti-tracking cell) + DK-8 vae (positive control cohort).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DIRS="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0,${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
CKPT_INPUT="${1:-${DEFAULT_DIRS}}"
OUT_DIR="${2:-${REPO}/logs/phaseB/e8}"

N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-200}"
HORIZONS="${HORIZONS:-1 3 5 10}"
DEVICE="${DEVICE:-cuda}"
ONLY_PATTERN="${ONLY_PATTERN:-*_best_model.pt}"
INCLUDE_REGEX="${INCLUDE_REGEX:-(sweep_doorkey16_v6|sweep_dk8_vae|rigor_dk8_vae|sweep_doorkey16_v5dc|sweep_dk8_v6)}"

mkdir -p "${OUT_DIR}"

IFS=',' read -ra DIRS <<< "${CKPT_INPUT}"
CKPTS=()
for d in "${DIRS[@]}"; do
  [[ -d "$d" ]] || { echo "skip missing $d" >&2; continue; }
  while IFS= read -r f; do
    if [[ -n "${INCLUDE_REGEX}" ]]; then
      if [[ "$(basename "$f")" =~ ${INCLUDE_REGEX} ]]; then CKPTS+=("$f"); fi
    else
      CKPTS+=("$f")
    fi
  done < <(find "$d" -type f -name "${ONLY_PATTERN}" | sort)
done

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "no ckpts matched" >&2; exit 2
fi

cd "${REPO}/discrete_mbrl"
echo "Phase B / E8: ${#CKPTS[@]} checkpoints"
echo "  n_episodes=${N_EPISODES}  max_steps=${MAX_STEPS}  horizons=${HORIZONS}  device=${DEVICE}"

for ckpt in "${CKPTS[@]}"; do
  rel="${ckpt#${REPO}/}"
  name="${rel//\//__}"; name="${name%.pt}"
  out="${OUT_DIR}/${name}.json"; log="${OUT_DIR}/${name}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${name}"; continue; fi
  echo "[run]  ${name}"
  set +e
  python analyze_wm_phaseB_e8.py \
    --model_path "${ckpt}" \
    --n_episodes "${N_EPISODES}" --max_steps "${MAX_STEPS}" \
    --horizons ${HORIZONS} --device "${DEVICE}" \
    --output_json "${out}" > "${log}" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[fail] ${name} (exit ${rc})"; rm -f "${out}"
  else
    echo "[ok]   ${name}"
  fi
done
