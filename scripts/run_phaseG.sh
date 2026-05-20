#!/usr/bin/env bash
# Phase G: train multi-step-loss WMs on a frozen encoder, then re-run all the
# diagnostics (Phase A analyzer, Phase D planning, Phase J value-coherence).
# VAE-only first round; 3 source ckpts.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseG"
mkdir -p "${OUT}/train" "${OUT}/analyzer" "${OUT}/planning" "${OUT}/jvalue"

K="${K:-5}"
EPOCHS="${EPOCHS:-100}"

SOURCES=(
  "sweep_dk8_vae_s1_best_model"
  "sweep_dk8_vae_s2_best_model"
  "sweep_dk8_vae_s3_best_model"
)

cd "${REPO}/discrete_mbrl"

for src in "${SOURCES[@]}"; do
  src_path="${CKPT_DIR}/${src}.pt"
  ms_path="${CKPT_DIR}/multistep_${src}.pt"
  if [[ ! -f "${src_path}" ]]; then echo "[skip-miss] ${src}"; continue; fi

  # Train multi-step WM
  if [[ ! -s "${ms_path}" ]]; then
    echo "[train] ${src} (K=${K})"
    set +e
    python train_multistep_oracle_wm.py \
      --model_path "${src_path}" --output_path "${ms_path}" \
      --K "${K}" --epochs "${EPOCHS}" \
      --n_random_eps 300 --n_policy_eps 300 \
      --batch_size 256 --lr 1e-4 \
      --device cuda > "${OUT}/train/multistep_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-train] ${src}"; continue; fi
  else
    echo "[skip-train] ${src} (already exists)"
  fi

  # Phase A analyzer (per-class WM accuracy)
  ana_out="${OUT}/analyzer/multistep_${src}.json"
  if [[ ! -s "${ana_out}" ]]; then
    echo "[analyze]  ${src}"
    set +e
    python analyze_wm_multistep.py \
      --model_path "${ms_path}" \
      --n_rollouts 300 --probe_frames 10000 \
      --horizons 1 3 5 10 \
      --device cuda --output_json "${ana_out}" > "${OUT}/analyzer/multistep_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-ana] ${src}"; rm -f "${ana_out}"; fi
  fi

  # Phase D planning (rank correlation)
  plan_out="${OUT}/planning/multistep_${src}.json"
  if [[ ! -s "${plan_out}" ]]; then
    echo "[plan]     ${src}"
    set +e
    python analyze_wm_planning.py \
      --model_path "${ms_path}" \
      --n_episodes 300 --max_steps 200 \
      --horizons 5 10 20 --stochastic \
      --device cuda --output_json "${plan_out}" > "${OUT}/planning/multistep_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-plan] ${src}"; rm -f "${plan_out}"; fi
  fi

  # Phase J value coherence
  jval_out="${OUT}/jvalue/multistep_${src}.json"
  if [[ ! -s "${jval_out}" ]]; then
    echo "[jvalue]   ${src}"
    set +e
    python analyze_wm_phaseB_e8.py \
      --model_path "${ms_path}" \
      --n_episodes 50 --max_steps 150 \
      --horizons 1 2 3 4 5 6 7 8 9 10 \
      --device cuda --output_json "${jval_out}" > "${OUT}/jvalue/multistep_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-jval] ${src}"; rm -f "${jval_out}"; fi
  fi

  echo "[ok] ${src}"
done
