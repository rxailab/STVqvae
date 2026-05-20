#!/usr/bin/env bash
# Phase VIII-A: replay diagnostic.
#
# Take the s2 (SUCCEEDED) and s5 (FAILED) ε=0.50 best_model encoders. Freeze.
# Collect 600 random-action episodes (no policy mix) on DK-8 → fresh WM
# training buffer. Train an oracle WM offline (same protocol as Phase II
# Cell A from May). Analyse per-class WM_1.
#
# Predictions:
#   s2-oracle door ≥ 0.5 AND s5-oracle door ≥ 0.5 → buffer/data is fine; the
#       joint encoder/WM training is the issue. Push Phase VIII-B.
#   s5-oracle door < 0.5 → s5's encoder is degraded; the failure is in
#       encoder representations. Reframe.
#   Both fail → unexpected.
#
# Each oracle training ≈ 20 min on H200 + 5 min analysis = 25 min/job.
set -euo pipefail
REPO="/mmfs1/storage/users/xiar3/exp/STVqvae"
LOG_DIR="${REPO}/logs/phaseVIII_A"
mkdir -p "${LOG_DIR}"

DK8_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
PHVIII_DIR="${DK8_DIR}/phaseVIII_A"
mkdir -p "${PHVIII_DIR}"

cd "${REPO}/discrete_mbrl"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="${REPO}/.mplconfig"

declare -a SRC_CKPTS=(
  "s2_succeeded   ${DK8_DIR}/v6_eps050_s2_best_model.pt"
  "s5_failed      ${DK8_DIR}/v6_eps050_s5_best_model.pt"
)

for entry in "${SRC_CKPTS[@]}"; do
  label="${entry%% *}"
  src="${entry##* }"
  dst="${PHVIII_DIR}/oracle_randonly_${label}.pt"
  log="${LOG_DIR}/train_${label}.log"

  if [[ -s "${dst}" ]]; then
    echo "[skip-train] ${label}"
  else
    echo "[train-oracle] ${label}  src=${src}"
    set +e
    python train_oracle_wm.py \
      --model_path "${src}" \
      --output_path "${dst}" \
      --n_random_eps 600 --n_policy_eps 0 \
      --max_steps 200 \
      --epochs 100 \
      --batch_size 512 \
      --lr 1e-4 \
      --device cuda > "${log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-train] ${label} (rc=$rc)"; rm -f "${dst}"; continue; fi
  fi

  # Analyse: per-class WM + action_cond
  for analyzer in analyze_wm_semantic_accuracy analyze_wm_action_cond_v2; do
    out="${LOG_DIR}/${analyzer}_${label}.json"
    aLog="${LOG_DIR}/${analyzer}_${label}.log"
    if [[ -s "${out}" ]]; then continue; fi
    echo "[analyze] ${label} ${analyzer}"
    set +e
    case "${analyzer}" in
      analyze_wm_semantic_accuracy)
        python ${analyzer}.py --model_path "${dst}" --n_frames 5000 --device cuda --output_json "${out}" > "${aLog}" 2>&1 ;;
      analyze_wm_action_cond_v2)
        python ${analyzer}.py --model_path "${dst}" --n_states 2000 --device cuda --output_json "${out}" > "${aLog}" 2>&1 ;;
    esac
  done
done

echo "Done."
