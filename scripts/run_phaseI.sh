#!/usr/bin/env bash
# Phase I: head calibration on the imagined-rollout distribution.
# For each source ckpt, re-train ONLY the reward + discount heads on (ẑ_k, a_k,
# r_k_real, γ_k_real) tuples, then re-run the planning analyzer.
# Targets: 5 oracle ckpts (the cohort F-light just measured).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
TRAIN_LOG_DIR="${REPO}/logs/phaseI/train"
PLAN_DIR="${REPO}/logs/phaseI/planning"
mkdir -p "${TRAIN_LOG_DIR}" "${PLAN_DIR}"

CKPTS=(
  "${CKPT_DIR}/oracle_sweep_dk8_v6_s4_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_v6_s5_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_v5dc_s4_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_vae_s1_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_vae_s2_best_model.pt"
)

cd "${REPO}/discrete_mbrl"

for src in "${CKPTS[@]}"; do
  if [[ ! -f "${src}" ]]; then echo "[skip-miss] $(basename ${src})"; continue; fi
  base="$(basename "${src}" .pt)"
  calib="${CKPT_DIR}/calib_${base}.pt"
  plan_out="${PLAN_DIR}/calib_${base}.json"
  train_log="${TRAIN_LOG_DIR}/calib_${base}.log"
  plan_log="${PLAN_DIR}/calib_${base}.log"

  if [[ -s "${plan_out}" ]]; then echo "[skip] ${base}"; continue; fi

  echo "[calib] ${base}"
  set +e
  python calibrate_heads_phaseI.py \
    --model_path "${src}" --output_path "${calib}" \
    --n_episodes 150 --max_steps 200 --K 10 \
    --epochs 50 --batch_size 512 --lr 1e-3 \
    --device cuda > "${train_log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-calib] ${base}"; continue; fi

  echo "[plan]  ${base}"
  set +e
  python analyze_wm_planning.py \
    --model_path "${calib}" \
    --n_episodes 100 --max_steps 150 --horizons 5 10 20 \
    --device cuda --output_json "${plan_out}" > "${plan_log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-plan] ${base}"; rm -f "${plan_out}"; else echo "[ok] ${base}"; fi
done
