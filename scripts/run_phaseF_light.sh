#!/usr/bin/env bash
# Phase F-light: does the offline-trained oracle WM also fix the planning failures
# that the online-trained WM exhibits? Re-run the E8 (policy/value coherence) and
# Phase D (imagined-vs-real return) analyzers on the existing oracle ckpts.
#
# The oracle ckpts share encoder + policy + critic with their source; only the
# transition_model differs. So a direct comparison of E8 / Phase D between
# oracle and source isolates the WM's planning quality.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORACLE_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
E8_OUT="${REPO}/logs/phaseF_light/e8"
D_OUT="${REPO}/logs/phaseF_light/planning"

mkdir -p "${E8_OUT}" "${D_OUT}"

mapfile -t ORACLE_CKPTS < <(find "${ORACLE_DIR}" -maxdepth 1 -name 'oracle_*.pt' | sort)

cd "${REPO}/discrete_mbrl"
echo "Phase F-light on ${#ORACLE_CKPTS[@]} oracle ckpts"

for src in "${ORACLE_CKPTS[@]}"; do
  name="$(basename "${src}" .pt)"

  # E8 — policy/value coherence under the trained policy
  e8_out="${E8_OUT}/${name}.json"
  e8_log="${E8_OUT}/${name}.log"
  if [[ -s "${e8_out}" ]]; then
    echo "[skip-e8] ${name}"
  else
    echo "[E8] ${name}"
    set +e
    python analyze_wm_phaseB_e8.py \
      --model_path "${src}" \
      --n_episodes 50 --max_steps 150 \
      --horizons 1 5 10 \
      --device cuda \
      --output_json "${e8_out}" > "${e8_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-e8] ${name}"; rm -f "${e8_out}"; fi
  fi

  # Phase D — imagined vs real return
  d_out="${D_OUT}/${name}.json"
  d_log="${D_OUT}/${name}.log"
  if [[ -s "${d_out}" ]]; then
    echo "[skip-D] ${name}"
  else
    echo "[D]  ${name}"
    set +e
    python analyze_wm_planning.py \
      --model_path "${src}" \
      --n_episodes 100 --max_steps 150 \
      --horizons 5 10 20 \
      --device cuda \
      --output_json "${d_out}" > "${d_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-D] ${name}"; rm -f "${d_out}"; fi
  fi
done
