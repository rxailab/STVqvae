#!/usr/bin/env bash
# Phase H: longer multi-step training (K=10, 200 epochs) WITH value-alignment loss.
# Tests whether training the WM to preserve V_eta(z) along free-running rollouts
# closes the planning rank-coherence gap.
# 5 VAE source ckpts (entire DK-8 vae cohort with policies that work).

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseH"
mkdir -p "${OUT}/train" "${OUT}/analyzer" "${OUT}/planning" "${OUT}/jvalue"

K="${K:-10}"
EPOCHS="${EPOCHS:-200}"
VAL_COEF="${VAL_COEF:-1.0}"

SOURCES=(
  "sweep_dk8_vae_s1_best_model"
  "sweep_dk8_vae_s2_best_model"
  "sweep_dk8_vae_s3_best_model"
  "sweep_dk8_vae_s4_best_model"
  "sweep_dk8_vae_s5_best_model"
)

cd "${REPO}/discrete_mbrl"

for src in "${SOURCES[@]}"; do
  src_path="${CKPT_DIR}/${src}.pt"
  ms_path="${CKPT_DIR}/phaseH_${src}.pt"
  if [[ ! -f "${src_path}" ]]; then echo "[skip-miss] ${src}"; continue; fi

  if [[ ! -s "${ms_path}" ]]; then
    echo "[train] ${src}  K=${K}  epochs=${EPOCHS}  val_coef=${VAL_COEF}"
    set +e
    python train_multistep_oracle_wm.py \
      --model_path "${src_path}" --output_path "${ms_path}" \
      --K "${K}" --epochs "${EPOCHS}" \
      --n_random_eps 400 --n_policy_eps 400 \
      --batch_size 256 --lr 1e-4 \
      --value_align_coef "${VAL_COEF}" \
      --device cuda > "${OUT}/train/phaseH_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-train] ${src}"; continue; fi
  else
    echo "[skip-train] ${src}"
  fi

  # Diagnostics
  ana_out="${OUT}/analyzer/phaseH_${src}.json"
  if [[ ! -s "${ana_out}" ]]; then
    set +e
    python analyze_wm_multistep.py \
      --model_path "${ms_path}" \
      --n_rollouts 300 --probe_frames 10000 \
      --horizons 1 3 5 10 \
      --device cuda --output_json "${ana_out}" > "${OUT}/analyzer/phaseH_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then rm -f "${ana_out}"; fi
  fi

  plan_out="${OUT}/planning/phaseH_${src}.json"
  if [[ ! -s "${plan_out}" ]]; then
    set +e
    python analyze_wm_planning.py \
      --model_path "${ms_path}" \
      --n_episodes 300 --max_steps 200 \
      --horizons 5 10 20 --stochastic \
      --device cuda --output_json "${plan_out}" > "${OUT}/planning/phaseH_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then rm -f "${plan_out}"; fi
  fi

  jval_out="${OUT}/jvalue/phaseH_${src}.json"
  if [[ ! -s "${jval_out}" ]]; then
    set +e
    python analyze_wm_phaseB_e8.py \
      --model_path "${ms_path}" \
      --n_episodes 50 --max_steps 150 \
      --horizons 1 2 3 4 5 6 7 8 9 10 \
      --device cuda --output_json "${jval_out}" > "${OUT}/jvalue/phaseH_${src}.log" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then rm -f "${jval_out}"; fi
  fi

  echo "[ok] ${src}"
done
