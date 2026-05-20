#!/usr/bin/env bash
# Phase V: extend Phase T's Dyna comparison to vae_s2 and vae_s3 source encoders.
#
# Phase T showed that on vae_s1 (action-blind starting point) λ=0.5 finetuning
# unlocked a +0.29 absolute Dyna improvement. Phase V tests whether this
# constructive recipe generalizes to vae_s2 and vae_s3 (which start partially
# action-aware). 2 src × 4 λ × 2 modes × 2 seeds = 32 Dyna runs.
#
# Hypotheses:
#   - Δ(oracle - vanilla) is positive at λ=0.5 on vae_s2 and vae_s3 too
#     ⇒ recipe generalizes ⇒ strongest paper claim
#   - Δ is regime-bounded (positive on vae_s2 like Phase O, but vae_s3 already
#     at ceiling so WM hurts) ⇒ ties cleanly to four-regime taxonomy
#   - Δ inconsistent ⇒ weakens constructive claim
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
RM_CKPTS="${CKPT_DIR}/phaseR_multi"
OUT="${REPO}/logs/phaseV"
DYNA_CKPTS="${CKPT_DIR}/phaseV"
mkdir -p "${OUT}" "${DYNA_CKPTS}"

LAMBDAS=(0.0 0.5 2.0 10.0)
SOURCES=("vae_s2" "vae_s3")

cd "${REPO}/discrete_mbrl"

run_dyna() {
  local src="$1" lam="$2" mode="$3" seed="$4"
  local enc_ckpt="${RM_CKPTS}/finetuned_${src}_lam${lam}.pt"
  local wm_ckpt="${RM_CKPTS}/oracle_finetuned_${src}_lam${lam}.pt"
  local out_ckpt="${DYNA_CKPTS}/dyna_${src}_lam${lam}_${mode}_s${seed}.pt"
  local log="${OUT}/dyna_${src}_lam${lam}_${mode}_s${seed}.log"
  if [[ -s "${out_ckpt}" ]]; then echo "[skip] ${src} lam=${lam} ${mode} s=${seed}"; return; fi
  if [[ ! -f "${enc_ckpt}" ]]; then echo "[skip-miss] enc ${src} lam=${lam}"; return; fi
  echo "[dyna-V] ${src} lam=${lam} mode=${mode} seed=${seed}"
  set +e
  PYTHONHASHSEED=${seed} python train_dyna.py \
    --src_ckpt "${enc_ckpt}" \
    --wm_ckpt  "${wm_ckpt}" \
    --wm_mode  "${mode}" \
    --output_path "${out_ckpt}" \
    --total_steps 300000 \
    --num_envs 16 --n_steps_per_update 128 \
    --imagine_K 5 --imagine_ratio 0.5 \
    --ppo_entropy_coef 0.02 \
    --lr 3e-4 --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${src} lam=${lam} ${mode} s=${seed}"; rm -f "${out_ckpt}"; fi
}

for src in "${SOURCES[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    WM="${RM_CKPTS}/oracle_finetuned_${src}_lam${lam}.pt"
    for seed in 1 2; do
      run_dyna "${src}" "${lam}" none   "${seed}"
      run_dyna "${src}" "${lam}" oracle "${seed}"
    done
  done
done
echo "Phase V complete."
