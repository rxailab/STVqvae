#!/usr/bin/env bash
# Caveat 3: MLP capacity sweep on the offline trainer.
# 3 source ckpts (one per encoder family) × 4 hidden widths = 12 oracle runs
# + 12 analyzer runs. Tests whether the gap-closure of Phase E2 is capacity-limited.
# Hypothesis: closure should be near-identical across capacities, since 256-256
# already extracts the encoder's information.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseE/arch"
TRAIN_LOG="${OUT}/train"
mkdir -p "${OUT}" "${TRAIN_LOG}"

SOURCES=(
  "sweep_dk8_v6_s4_best_model"      # v6 family
  "sweep_dk8_v5dc_s4_best_model"    # v5dc family
  "sweep_dk8_vae_s1_best_model"     # vae family
)
HIDDENS=(128 256 512 1024)

cd "${REPO}/discrete_mbrl"

for src in "${SOURCES[@]}"; do
  src_path="${CKPT_DIR}/${src}.pt"
  if [[ ! -f "${src_path}" ]]; then echo "[skip-miss] ${src}"; continue; fi

  for h in "${HIDDENS[@]}"; do
    oracle="${CKPT_DIR}/oracle_${src}_h${h}.pt"
    out="${OUT}/oracle_${src}_h${h}.json"
    train_log="${TRAIN_LOG}/oracle_${src}_h${h}_train.log"
    ana_log="${OUT}/oracle_${src}_h${h}.log"

    if [[ -s "${out}" ]]; then echo "[skip] ${src} h=${h}"; continue; fi

    echo "[train] ${src} hidden=${h}"
    set +e
    python train_oracle_wm.py \
      --model_path "${src_path}" --output_path "${oracle}" \
      --n_random_eps 300 --n_policy_eps 300 \
      --epochs 100 --batch_size 512 --lr 1e-4 \
      --trans_hidden "${h}" --trans_depth 3 \
      --device cuda > "${train_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-train] ${src} h=${h}"; continue; fi

    echo "[analyze] ${src} h=${h}"
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
