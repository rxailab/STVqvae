#!/usr/bin/env bash
# Caveat 3 (deeper): Spatial ConvNet transition model — different architecture
# class entirely. 3 VAE source ckpts × 2 conv hidden widths = 6 oracle runs.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseE/conv"
TRAIN_LOG="${OUT}/train"
mkdir -p "${OUT}" "${TRAIN_LOG}"

SOURCES=(
  "sweep_dk8_vae_s1_best_model"
  "sweep_dk8_vae_s2_best_model"
  "sweep_dk8_vae_s3_best_model"
)
CONV_HIDDENS=(32 64)

cd "${REPO}/discrete_mbrl"

for src in "${SOURCES[@]}"; do
  src_path="${CKPT_DIR}/${src}.pt"
  if [[ ! -f "${src_path}" ]]; then echo "[skip-miss] ${src}"; continue; fi
  for h in "${CONV_HIDDENS[@]}"; do
    oracle="${CKPT_DIR}/conv_oracle_${src}_h${h}.pt"
    out="${OUT}/conv_oracle_${src}_h${h}.json"
    train_log="${TRAIN_LOG}/conv_oracle_${src}_h${h}_train.log"
    ana_log="${OUT}/conv_oracle_${src}_h${h}.log"
    if [[ -s "${out}" ]]; then echo "[skip] ${src} h=${h}"; continue; fi

    echo "[train-conv] ${src} hidden=${h}"
    set +e
    python train_conv_oracle_wm.py \
      --model_path "${src_path}" --output_path "${oracle}" \
      --n_random_eps 300 --n_policy_eps 300 \
      --epochs 100 --batch_size 256 --lr 1e-4 \
      --conv_hidden "${h}" --conv_depth 3 \
      --device cuda > "${train_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-train] ${src} h=${h}"; continue; fi

    echo "[analyze]    ${src} h=${h}"
    set +e
    python analyze_wm_multistep.py \
      --model_path "${oracle}" \
      --n_rollouts 300 --probe_frames 10000 \
      --horizons 1 3 5 10 \
      --device cuda --output_json "${out}" > "${ana_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-analyze] ${src} h=${h}"; rm -f "${out}"; else echo "[ok] ${src} h=${h}"; fi
  done
done
