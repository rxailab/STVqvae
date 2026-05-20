#!/usr/bin/env bash
# Phase I-v2: re-evaluate planning rank_corr with stochastic policy + 300 episodes
# to fix the N=3 limitation in the deterministic-argmax run.
#
# Three conditions evaluated on 5 source-encoder ckpts each (15 runs total):
#   online  : original ckpts as shipped (no oracle, no calibration)
#   oracle  : Phase E2 oracle WMs (offline-trained transition model)
#   calib   : Phase I calibrated heads on top of oracle

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseI_v2"
mkdir -p "${OUT}/online" "${OUT}/oracle" "${OUT}/calib"

N_EPISODES="${N_EPISODES:-300}"
MAX_STEPS="${MAX_STEPS:-200}"
HORIZONS="${HORIZONS:-5 10 20}"

SOURCES=(
  "sweep_dk8_v6_s4_best_model"
  "sweep_dk8_v6_s5_best_model"
  "sweep_dk8_v5dc_s4_best_model"
  "sweep_dk8_vae_s1_best_model"
  "sweep_dk8_vae_s2_best_model"
)

cd "${REPO}/discrete_mbrl"

run_one() {
  local label="$1" path="$2" base="$3"
  local out="${OUT}/${label}/${base}.json" log="${OUT}/${label}/${base}.log"
  if [[ ! -f "${path}" ]]; then echo "[skip-miss] ${label}/${base}"; return; fi
  if [[ -s "${out}" ]]; then echo "[skip] ${label}/${base}"; return; fi
  echo "[run-${label}] ${base}"
  set +e
  python analyze_wm_planning.py \
    --model_path "${path}" \
    --n_episodes "${N_EPISODES}" --max_steps "${MAX_STEPS}" \
    --horizons ${HORIZONS} --stochastic \
    --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${label}/${base}"; rm -f "${out}"; else echo "[ok] ${label}/${base}"; fi
}

for src in "${SOURCES[@]}"; do
  run_one online  "${CKPT_DIR}/${src}.pt"            "${src}"
  run_one oracle  "${CKPT_DIR}/oracle_${src}.pt"     "${src}"
  run_one calib   "${CKPT_DIR}/calib_oracle_${src}.pt" "${src}"
done
