#!/usr/bin/env bash
# Phase E2 camera-ready tightening:
#   (a) expansion: 5 more source ckpts with mixed (random+policy) buffer
#   (b) random-only-buffer control: rerun original 5 with n_policy_eps=0
# Both produce oracle ckpts named oracle_<src>{,_random}.pt and analyzer JSONs.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORACLE_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
ANALYSIS_DIR="${REPO}/logs/phaseE/oracle_analyzer"
TRAIN_LOG_DIR="${REPO}/logs/phaseE/oracle_train"
EPOCHS=100
BATCH=512
LR=1e-4
N_ROLLOUTS=300
PROBE_FRAMES=10000
HORIZONS="1 3 5 10"
DEVICE=cuda

mkdir -p "${ANALYSIS_DIR}" "${TRAIN_LOG_DIR}"

# (a) Expansion ckpts (mixed buffer, same recipe as the original 5)
EXPAND_CKPTS=(
  "${ORACLE_DIR}/sweep_doorkey8_v6_s1_best_model.pt"
  "${ORACLE_DIR}/sweep_doorkey8_v6_s2_best_model.pt"
  "${ORACLE_DIR}/sweep_doorkey8_v6_s3_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_vae_s3_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_v5dc_s5_best_model.pt"
)

# (b) Random-only-buffer control: rerun original 5 with n_random_eps=600, n_policy_eps=0
RANDOM_CKPTS=(
  "${ORACLE_DIR}/sweep_dk8_v6_s4_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_v6_s5_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_vae_s1_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_vae_s2_best_model.pt"
  "${ORACLE_DIR}/sweep_dk8_v5dc_s4_best_model.pt"
)

cd "${REPO}/discrete_mbrl"

train_one() {
  local src="$1" suffix="$2" n_rand="$3" n_pol="$4"
  if [[ ! -f "${src}" ]]; then echo "[skip] missing ${src}"; return; fi
  local base="$(basename "${src}" .pt)"
  local oracle="${ORACLE_DIR}/oracle_${base}${suffix}.pt"
  local out="${ANALYSIS_DIR}/oracle_${base}${suffix}.json"
  local train_log="${TRAIN_LOG_DIR}/oracle_${base}${suffix}_train.log"
  local ana_log="${ANALYSIS_DIR}/oracle_${base}${suffix}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${base}${suffix}"; return; fi

  echo "[train] ${base}${suffix}  (n_rand=${n_rand}, n_pol=${n_pol})"
  set +e
  python train_oracle_wm.py \
    --model_path "${src}" --output_path "${oracle}" \
    --n_random_eps "${n_rand}" --n_policy_eps "${n_pol}" \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr ${LR} \
    --device ${DEVICE} > "${train_log}" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-train] ${base}${suffix}"; return; fi

  echo "[analyze] ${base}${suffix}"
  set +e
  python analyze_wm_multistep.py \
    --model_path "${oracle}" \
    --n_rollouts ${N_ROLLOUTS} --probe_frames ${PROBE_FRAMES} \
    --horizons ${HORIZONS} --device ${DEVICE} \
    --output_json "${out}" > "${ana_log}" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-analyze] ${base}${suffix}"; rm -f "${out}"; else echo "[ok]   ${base}${suffix}"; fi
}

echo "=== (a) expansion: ${#EXPAND_CKPTS[@]} ckpts, mixed buffer ==="
for src in "${EXPAND_CKPTS[@]}"; do train_one "${src}" "" 300 300; done

echo
echo "=== (b) random-only-buffer control: ${#RANDOM_CKPTS[@]} ckpts ==="
for src in "${RANDOM_CKPTS[@]}"; do train_one "${src}" "_randonly" 600 0; done
