#!/usr/bin/env bash
# Phase TX: test the probe / action_dep trade-off on transformer transition
# architectures. Two variants:
#   * "transformer"     — full encoder-decoder TransformerTransitionModel
#                         (nhead=8, layers=6+6, dim_ff=1024)
#   * "transformerdec"  — causal decoder-only TransformerDecTransitionModel
#                         (nhead=4, layers=6, dim_ff=256; IRIS-style)
#
# For each variant: train 2 fresh VQ-VAE + transformer-T encoders on DK-8,
# then run the same Phase R lambda-sweep used everywhere else, then Phase A
# (probe + WM) and Phase N (action_dep_ratio). The diagnostic oracle WM
# stays MLP — that is the protocol used for the existing 72-cell main pool,
# so transformer-T cells are directly comparable.
#
# Per encoder: ~2h training + 4 cells * ~30min sweep = ~4h.
# Two variants * 2 seeds = 4 encoders. 4 * ~4h = ~16h.
#
# Idempotent: skip-guards on encoder ckpts and per-cell logs allow resume.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseTX"
mkdir -p "${LOG_DIR}"

LAMBDAS=(0.0 0.5 2.0 10.0)
ENV_NAME="MiniGrid-DoorKey-8x8-v0"
MODEL_DIR="${REPO}/discrete_mbrl/model_free/models/${ENV_NAME}"
PR_DIR="${MODEL_DIR}/phaseTX"
mkdir -p "${PR_DIR}"

# ── Step 1: Train fresh VQ-VAE + transformer-T encoders ───────────────────
train_encoder() {
  local trans_kind="$1" seed="$2" run_name="$3"
  local target_ckpt="${MODEL_DIR}/${run_name}_best_model.pt"
  if [[ -s "${target_ckpt}" ]]; then echo "[skip-train] ${run_name}"; return; fi

  echo "[train] ${run_name} trans=${trans_kind} seed=${seed}"
  cd "${REPO}/discrete_mbrl/model_free"
  set +e
  # VQ-VAE encoder + the transformer-family transition model. Mirrors the
  # PPO + e2e + WM-aux config used for the main DK-8 VQ runs in Phase A.
  python -u train.py \
    --env_name "${ENV_NAME}" \
    --ae_model_type vqvae --ae_model_version 6 \
    --embedding_dim 64 --filter_size 8 --codebook_size 64 \
    --trans_model_type "${trans_kind}" --trans_model_version 1 \
    --seed "${seed}" \
    --mf_steps 5000000 \
    --batch_size 4096 \
    --num_envs 16 \
    --ppo_iters 10 \
    --ppo_batch_size 64 \
    --ppo_entropy_coef 0.01 \
    --ppo_gae_lambda 0.95 \
    --ppo_norm_advantages \
    --ppo_max_grad_norm 0.5 \
    --learning_rate 1e-4 \
    --e2e_loss \
    --encoder_lr 1e-5 \
    --encoder_lr_cosine \
    --encoder_snapback \
    --snapback_threshold 0.5 \
    --snapback_patience 100 \
    --snapback_min_reward 0.0 \
    --ortho_init \
    --model_dir .. \
    --run_name "${run_name}" \
    --device cuda \
    --save \
    --use_world_model \
    --wm_aux_coef 0.1 \
    --wm_train_freq 1 \
    > "${LOG_DIR}/train_${run_name}.log" 2>&1
  rc=$?; set -e
  cd "${REPO}"
  if [[ $rc -ne 0 ]]; then echo "[fail-train] ${run_name}"; return 1; fi
  echo "[ok-train] ${run_name}"
}

# ── Step 2: Run lambda-sweep pipeline per (encoder, lambda) ──────────────
run_pipeline() {
  local label="$1" src_ckpt="$2" lam="$3"
  local enc_ckpt="${PR_DIR}/finetuned_${label}_lam${lam}.pt"
  local oracle_ckpt="${PR_DIR}/oracle_finetuned_${label}_lam${lam}.pt"
  local phaseA_out="${LOG_DIR}/phaseA_${label}_lam${lam}.json"
  local phaseN_out="${LOG_DIR}/phaseN_${label}_lam${lam}.json"
  local ft_log="${LOG_DIR}/finetune_${label}_lam${lam}.log"
  local oracle_log="${LOG_DIR}/oracle_${label}_lam${lam}.log"
  local A_log="${LOG_DIR}/phaseA_${label}_lam${lam}.log"
  local N_log="${LOG_DIR}/phaseN_${label}_lam${lam}.log"

  cd "${REPO}/discrete_mbrl"

  if [[ -s "${enc_ckpt}" ]]; then echo "[skip] finetune ${label} lam=${lam}"; else
    echo "[finetune] ${label} lam=${lam}"
    set +e
    python finetune_encoder_action_cond.py \
      --model_path "${src_ckpt}" --output_path "${enc_ckpt}" \
      --action_aux_weight "${lam}" --epochs 10 \
      --n_random_eps 200 --n_policy_eps 200 \
      --batch_size 128 --lr 1e-4 \
      --device cuda > "${ft_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-ft] ${label} lam=${lam}"; rm -f "${enc_ckpt}"; return; fi
  fi

  if [[ -s "${oracle_ckpt}" ]]; then echo "[skip] oracle ${label} lam=${lam}"; else
    echo "[oracle] ${label} lam=${lam}"
    set +e
    python train_oracle_wm.py \
      --model_path "${enc_ckpt}" --output_path "${oracle_ckpt}" \
      --n_random_eps 250 --n_policy_eps 250 \
      --epochs 50 --batch_size 256 --lr 1e-4 \
      --device cuda > "${oracle_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-oracle] ${label} lam=${lam}"; rm -f "${oracle_ckpt}"; return; fi
  fi

  if [[ ! -s "${phaseA_out}" ]]; then
    echo "[phaseA] ${label} lam=${lam}"
    set +e
    python analyze_wm_multistep.py --model_path "${oracle_ckpt}" \
      --n_rollouts 300 --probe_frames 10000 --horizons 1 3 5 10 \
      --device cuda --output_json "${phaseA_out}" > "${A_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-A] ${label} lam=${lam}"; rm -f "${phaseA_out}"; fi
  fi

  if [[ ! -s "${phaseN_out}" ]]; then
    echo "[phaseN] ${label} lam=${lam}"
    set +e
    python analyze_wm_action_cond.py --model_path "${oracle_ckpt}" \
      --n_states 2000 --device cuda \
      --output_json "${phaseN_out}" > "${N_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-N] ${label} lam=${lam}"; rm -f "${phaseN_out}"; fi
  fi
  echo "[ok] ${label} lam=${lam}"
}

# ── Configuration: 2 transformer variants x 2 seeds = 4 encoders ─────────
declare -a CONFIGS=(
  "transformer|tx_enc_s1|phaseTX_enc_s1|1"
  "transformer|tx_enc_s2|phaseTX_enc_s2|2"
  "transformerdec|tx_dec_s1|phaseTX_dec_s1|1"
  "transformerdec|tx_dec_s2|phaseTX_dec_s2|2"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r trans_kind label run_name seed <<< "${entry}"
  echo "================================================"
  echo "Variant: trans=${trans_kind}  label=${label}  seed=${seed}"
  echo "================================================"

  if ! train_encoder "${trans_kind}" "${seed}" "${run_name}"; then
    echo "[skip-cfg] training failed for ${label}"; continue
  fi

  src_ckpt="${MODEL_DIR}/${run_name}_best_model.pt"
  if [[ ! -f "${src_ckpt}" ]]; then echo "[skip-miss] ${src_ckpt}"; continue; fi

  for lam in "${LAMBDAS[@]}"; do
    run_pipeline "${label}" "${src_ckpt}" "${lam}"
  done
done

echo "Phase TX complete."
