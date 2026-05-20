#!/usr/bin/env bash
# Phase II Cell D: action-uniform oracle WM training.
# Two modes per source ckpt:
#   F1a (mixed_step): per-step ε-greedy mix of policy + random actions
#   F1b (branch):     env-clone each policy state with every non-policy action
#
# Goal: test whether restoring action diversity (without going back to a pure
# random buffer) closes the per-class WM gap.
#
# 6 source ckpts × 2 modes = 12 trainings + 24 analyses ≈ 5 h.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseII_cellD"
mkdir -p "${LOG_DIR}"

DK8="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
DK16="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
OUT_DK8="${DK8}/phaseII"
OUT_DK16="${DK16}/phaseII"
mkdir -p "${OUT_DK8}" "${OUT_DK16}"

cd "${REPO}/discrete_mbrl"

SRC_CKPTS=(
  "v6_s4_dk8     ${DK8}/sweep_dk8_v6_s4_best_model.pt          ${OUT_DK8}"
  "v6_s5_dk8     ${DK8}/sweep_dk8_v6_s5_best_model.pt          ${OUT_DK8}"
  "vae_s2_dk8    ${DK8}/sweep_dk8_vae_s2_best_model.pt         ${OUT_DK8}"
  "vae_s5_dk8    ${DK8}/sweep_dk8_vae_s5_best_model.pt         ${OUT_DK8}"
  "v5dc_s1_dk16  ${DK16}/sweep_doorkey16_v5dc_s1_best_model.pt ${OUT_DK16}"
  "v5dc_s2_dk16  ${DK16}/sweep_doorkey16_v5dc_s2_best_model.pt ${OUT_DK16}"
)

train_one() {
  local label="$1" src="$2" outdir="$3" mode="$4" suffix="$5"
  local dst="${outdir}/oracle_${suffix}_${label#*_}.pt"  # strip leading "label_"-style
  local dst_alt="${outdir}/oracle_${suffix}_${label}.pt"
  local log="${LOG_DIR}/${suffix}_${label}.log"

  if [[ -s "${dst_alt}" ]]; then echo "[skip] ${label} ${mode}"; return; fi
  if [[ ! -f "${src}" ]]; then echo "[miss-ckpt] ${src}"; return; fi

  echo "[${suffix}] ${label} -> ${dst_alt}"
  local extra=""
  if [[ "${mode}" == "mixed_step" ]]; then
    extra="--mix_prob 0.5 --n_episodes 600"
  else
    extra="--n_episodes 85"
  fi
  set +e
  python train_oracle_wm_actuniform.py \
    --model_path "${src}" \
    --output_path "${dst_alt}" \
    --mode "${mode}" \
    --max_steps 200 \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-4 \
    --device cuda ${extra} > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${label} ${mode} (rc=$rc)"; rm -f "${dst_alt}"; fi
}

analyze_one() {
  local label="$1" suffix="$2" outdir="$3"
  local dst="${outdir}/oracle_${suffix}_${label}.pt"
  if [[ ! -s "${dst}" ]]; then return; fi
  for analyzer in analyze_wm_semantic_accuracy analyze_wm_action_cond_v2; do
    local out="${LOG_DIR}/${analyzer}_${suffix}_${label}.json"
    local log="${LOG_DIR}/${analyzer}_${suffix}_${label}.log"
    if [[ -s "${out}" ]]; then echo "[skip-ana] ${label} ${suffix} ${analyzer}"; continue; fi
    echo "[analyze] ${label} ${suffix} ${analyzer}"
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
    if [[ $rc -ne 0 ]]; then echo "[fail-ana] ${label} ${suffix} (rc=$rc)"; fi
  done
}

# Training pass
for entry in "${SRC_CKPTS[@]}"; do
  read -r label src outdir <<< "${entry}"
  train_one "${label}" "${src}" "${outdir}" "mixed_step" "f1a_mixed"
  train_one "${label}" "${src}" "${outdir}" "branch"     "f1b_branch"
done

# Analysis pass
echo
echo "=== Analysing all Cell D ckpts ==="
for entry in "${SRC_CKPTS[@]}"; do
  read -r label src outdir <<< "${entry}"
  analyze_one "${label}" "f1a_mixed"  "${outdir}"
  analyze_one "${label}" "f1b_branch" "${outdir}"
done
echo "Done."
