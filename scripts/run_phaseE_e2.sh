#!/usr/bin/env bash
# Phase E2 driver: for each chosen ckpt, train an oracle WM offline-to-convergence
# on a frozen encoder, save the resulting ckpt, then run the full Phase A analyzer
# on it so the oracle WM result is directly comparable to the original online-WM
# result.
#
# Selection: 5 representative ckpts spanning v6 / vae / v5dc on DK-8.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORACLE_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
ANALYSIS_DIR="${1:-${REPO}/logs/phaseE/oracle_analyzer}"
TRAIN_LOG_DIR="${REPO}/logs/phaseE/oracle_train"
N_RANDOM="${N_RANDOM:-300}"
N_POLICY="${N_POLICY:-300}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-512}"
LR="${LR:-1e-4}"
N_ROLLOUTS="${N_ROLLOUTS:-300}"
PROBE_FRAMES="${PROBE_FRAMES:-10000}"
HORIZONS="${HORIZONS:-1 3 5 10}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${ANALYSIS_DIR}" "${TRAIN_LOG_DIR}"

CKPTS=(
  "${ORACLE_DIR}/sweep_dk8_v6_s4_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_v6_s5_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_vae_s1_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_vae_s2_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_v5dc_s4_best_model.pt"
)

cd "${REPO}/discrete_mbrl"

for src in "${CKPTS[@]}"; do
  if [[ ! -f "${src}" ]]; then echo "[skip] missing ${src}"; continue; fi
  base="$(basename "${src}" .pt)"
  oracle="${ORACLE_DIR}/oracle_${base}.pt"
  out="${ANALYSIS_DIR}/oracle_${base}.json"
  train_log="${TRAIN_LOG_DIR}/oracle_${base}_train.log"
  ana_log="${ANALYSIS_DIR}/oracle_${base}.log"

  if [[ -s "${out}" ]]; then echo "[skip] ${base}"; continue; fi

  echo "[train] oracle WM for ${base}"
  set +e
  python train_oracle_wm.py \
    --model_path  "${src}" \
    --output_path "${oracle}" \
    --n_random_eps "${N_RANDOM}" --n_policy_eps "${N_POLICY}" \
    --epochs "${EPOCHS}" --batch_size "${BATCH}" --lr "${LR}" \
    --device "${DEVICE}" > "${train_log}" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[fail-train] ${base} (exit ${rc}); see ${train_log}"
    continue
  fi

  echo "[analyze] ${base}"
  set +e
  python analyze_wm_multistep.py \
    --model_path  "${oracle}" \
    --n_rollouts  "${N_ROLLOUTS}" \
    --probe_frames "${PROBE_FRAMES}" \
    --horizons    ${HORIZONS} \
    --device      "${DEVICE}" \
    --output_json "${out}" > "${ana_log}" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[fail-analyze] ${base} (exit ${rc}); see ${ana_log}"
    rm -f "${out}"
  else
    echo "[ok]   ${base}"
  fi
done
