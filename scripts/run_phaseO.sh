#!/usr/bin/env bash
# Phase O: rerun Phase L's Dyna comparison with non-degenerate source encoders.
#
# Phase N showed action_dep_ratio = 0.0003 for online_vae_s1 vs 0.677 for
# oracle_vae_s2 vs 1.094 for conv_vae_s2_h64. If the encoder is the bottleneck,
# switching from vae_s1 to vae_s2 / vae_s3 should make oracle-Dyna actually
# beat vanilla PPO (Phase L showed they were noise-tied on vae_s1).
#
# Conditions: vanilla, oracle-Dyna, bootstrap-only-Dyna
# Encoders: vae_s2, vae_s3
# Seeds: 1, 2
# 3 × 2 × 2 = 12 runs at 300k steps ≈ 5 min each ≈ 1h total compute.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseO"
DYNA_CKPTS="${CKPT_DIR}/phaseO"
mkdir -p "${OUT}" "${DYNA_CKPTS}"

cd "${REPO}/discrete_mbrl"

run_dyna() {
  local enc="$1" mode="$2" wm_path="$3" seed="$4"
  local out_ckpt="${DYNA_CKPTS}/dyna_${enc}_${mode}_s${seed}.pt"
  local log="${OUT}/dyna_${enc}_${mode}_s${seed}.log"
  if [[ -s "${out_ckpt}" ]]; then echo "[skip] dyna_${enc}_${mode}_s${seed}"; return; fi
  echo "[dyna-O] enc=${enc} mode=${mode} seed=${seed}"
  set +e
  PYTHONHASHSEED=${seed} python train_dyna.py \
    --src_ckpt "${CKPT_DIR}/sweep_dk8_${enc}_best_model.pt" \
    --wm_ckpt  "${wm_path}" \
    --wm_mode  "${mode}" \
    --output_path "${out_ckpt}" \
    --total_steps 300000 \
    --num_envs 16 --n_steps_per_update 128 \
    --imagine_K 5 --imagine_ratio 0.5 \
    --ppo_entropy_coef 0.02 \
    --lr 3e-4 --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] dyna_${enc}_${mode}_s${seed}"; rm -f "${out_ckpt}"; else echo "[ok] dyna_${enc}_${mode}_s${seed}"; fi
}

for enc in vae_s2 vae_s3; do
  ORACLE="${CKPT_DIR}/oracle_sweep_dk8_${enc}_best_model.pt"
  if [[ ! -f "${ORACLE}" ]]; then echo "[skip-miss] oracle for ${enc}"; continue; fi
  for seed in 1 2; do
    # vanilla baseline (uses src encoder only, no WM signal)
    run_dyna "${enc}" none      ""         "${seed}"
    # standard Dyna with the (non-degenerate) oracle WM
    run_dyna "${enc}" oracle    "${ORACLE}" "${seed}"
    # bootstrap-only Dyna with the same WM
    run_dyna "${enc}" bootstrap "${ORACLE}" "${seed}"
  done
done
