#!/usr/bin/env bash
# Phase II Cell C: policy-buffer offline oracle WM training.
# Tests the prediction: training the WM on transitions the *policy* actually
# generates (which include door/key/goal interactions) closes the per-class
# WM gap that random-action buffers cannot.
#
# Source ckpts span both Phase I failure modes:
#   mode (b) — within-regime per-class:   v6 DK-8 s4, s5
#   mode (a) — drowned signal (VAE):      vae DK-8 s2, s5
#   mode (a) — extinct signal (v5dc DK16): v5dc DK-16 s1, s2
#
# Output: discrete_mbrl/.../phaseII/oracle_polonly_<stub>.pt
# ~15 min/ckpt on GPU. 6 source ckpts = ~90 min total.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseII_cellC"
mkdir -p "${LOG_DIR}"

DK8="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
DK16="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
OUT_DK8="${DK8}/phaseII"
OUT_DK16="${DK16}/phaseII"
mkdir -p "${OUT_DK8}" "${OUT_DK16}"

cd "${REPO}/discrete_mbrl"

SRC_CKPTS=(
  # mode (b): healthy v6 DK-8
  "v6_s4_dk8         ${DK8}/sweep_dk8_v6_s4_best_model.pt          ${OUT_DK8}/oracle_polonly_v6_s4.pt"
  "v6_s5_dk8         ${DK8}/sweep_dk8_v6_s5_best_model.pt          ${OUT_DK8}/oracle_polonly_v6_s5.pt"
  # mode (a) drowned: VAE DK-8
  "vae_s2_dk8        ${DK8}/sweep_dk8_vae_s2_best_model.pt         ${OUT_DK8}/oracle_polonly_vae_s2.pt"
  "vae_s5_dk8        ${DK8}/sweep_dk8_vae_s5_best_model.pt         ${OUT_DK8}/oracle_polonly_vae_s5.pt"
  # mode (a) extinct: v5dc DK-16
  "v5dc_s1_dk16      ${DK16}/sweep_doorkey16_v5dc_s1_best_model.pt ${OUT_DK16}/oracle_polonly_v5dc_s1.pt"
  "v5dc_s2_dk16      ${DK16}/sweep_doorkey16_v5dc_s2_best_model.pt ${OUT_DK16}/oracle_polonly_v5dc_s2.pt"
)

for entry in "${SRC_CKPTS[@]}"; do
  read -r label src dst <<< "${entry}"
  log="${LOG_DIR}/cellC_${label}.log"
  if [[ -s "${dst}" ]]; then echo "[skip] ${label}"; continue; fi
  if [[ ! -f "${src}" ]]; then echo "[miss-ckpt] ${src}"; continue; fi

  echo "[cellC] ${label} -> ${dst}"
  set +e
  python train_oracle_wm.py \
    --model_path "${src}" \
    --output_path "${dst}" \
    --n_random_eps 0 \
    --n_policy_eps 600 \
    --max_steps 200 \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-4 \
    --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${label} (rc=$rc)"; rm -f "${dst}"; fi
done

echo
echo "=== Phase II Cell C complete. Now re-analysing each ckpt ==="
for entry in "${SRC_CKPTS[@]}"; do
  read -r label src dst <<< "${entry}"
  if [[ ! -s "${dst}" ]]; then continue; fi
  for analyzer in analyze_wm_semantic_accuracy analyze_wm_action_cond_v2; do
    out="${LOG_DIR}/${analyzer}_${label}.json"
    log="${LOG_DIR}/${analyzer}_${label}.log"
    if [[ -s "${out}" ]]; then echo "[skip-analyze] ${label} ${analyzer}"; continue; fi
    echo "[analyze] ${label} ${analyzer}"
    set +e
    case "${analyzer}" in
      analyze_wm_semantic_accuracy)
        python ${analyzer}.py --model_path "${dst}" --n_frames 5000 \
          --device cuda --output_json "${out}" > "${log}" 2>&1
        ;;
      analyze_wm_action_cond_v2)
        python ${analyzer}.py --model_path "${dst}" --n_states 2000 \
          --device cuda --output_json "${out}" > "${log}" 2>&1
        ;;
    esac
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-analyze] ${label} (rc=$rc)"; fi
  done
done
echo "Done."
