#!/usr/bin/env bash
set -euo pipefail

# Multi-seed sweep v2 — adds v5dc (dead-code restart) and vae (continuous encoder
# + continuous MLP WM, positive control for probe↔WM dissociation).
#
# Calls analyze_wm_multistep.py after training (probe + multi-step WM in one pass).
# Writes per-checkpoint JSON to logs/wm_multistep/.
#
# Usage:
#   SEEDS="4 5" bash sweep_multiseed_v2.sh v2 v5dc v6        # W1 DK-8 extension
#   SEEDS="1 2 3 4 5" bash sweep_multiseed_v2.sh vae         # W2 VAE+WM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use main tree paths so dashboard, existing checkpoints, and sweep CSV are consistent
PROJECT_ROOT="/home/xiar3/experiments/STVqvae"
LOG_DIR="$PROJECT_ROOT/logs/sweep"
RESULTS_CSV="$LOG_DIR/sweep_results_v2.csv"
WM_OUT_DIR="$PROJECT_ROOT/logs/wm_multistep"
DEFAULT_LOCAL_PYTHON="/home/xiar3/experiments/miniforge3/envs/stvqvae/bin/python"
DEVICE="${DEVICE:-cuda}"

SEEDS="${SEEDS:-1 2 3}"
ENV_NAME="${ENV_NAME:-MiniGrid-DoorKey-8x8-v0}"
MF_STEPS="${MF_STEPS:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_ENVS="${NUM_ENVS:-16}"

if [[ $# -gt 0 ]]; then
    ENCODERS=("$@")
else
    ENCODERS=(v2 v5dc v6 vae)
fi

PYTHON_BIN="$DEFAULT_LOCAL_PYTHON"
[[ -x "$PYTHON_BIN" ]] || { echo "No Python at $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$WM_OUT_DIR" "$PROJECT_ROOT/.mplconfig"

if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "encoder,seed,env,best_reward,final_reward,overall_avg,probe_wall,probe_door,probe_key,probe_goal,probe_agent,wm_1_wall,wm_1_door,wm_1_key,wm_1_goal,wm_1_agent,wm_10_wall,wm_10_door,wm_10_key,wm_10_goal,wm_10_agent,r_1,r_5,r_10" > "$RESULTS_CSV"
fi

# ── Encoder-specific train args ──
get_train_args() {
    local enc="$1"
    case "$enc" in
        v2)
            echo "--ae_model_type vqvae --ae_model_version 2 --codebook_size 64 --embedding_dim 64 --filter_size 9 --trans_model_type discrete"
            ;;
        v5)
            echo "--ae_model_type vqvae --ae_model_version 5 --codebook_size 64 --embedding_dim 64 --filter_size 8 --trans_model_type discrete"
            ;;
        v5dc)
            echo "--ae_model_type vqvae --ae_model_version 5 --codebook_size 64 --embedding_dim 64 --filter_size 8 --dead_code_threshold 2.0 --trans_model_type discrete"
            ;;
        v6)
            echo "--ae_model_type vqvae --ae_model_version 6 --codebook_size 64 --embedding_dim 64 --filter_size 8 --trans_model_type discrete"
            ;;
        v9)
            echo "--ae_model_type vqvae --ae_model_version 9 --codebook_size 64 --embedding_dim 64 --filter_size 8 --trans_model_type discrete"
            ;;
        vae)
            # Spatial continuous VAE + continuous MLP transition (positive control).
            # sem_pre_vq and wm_aux_coef are OVERRIDDEN below to fit the continuous path.
            echo "--ae_model_type vae_spatial --ae_model_version 6 --embedding_dim 64 --filter_size 8 --trans_model_type continuous"
            ;;
        *)
            echo "Unknown encoder: $enc" >&2; exit 1
            ;;
    esac
}

env_short() {
    case "$1" in
        *DoorKey-8x8*) echo "dk8" ;;
        *DoorKey-16x16*) echo "dk16" ;;
        *) echo "env" ;;
    esac
}

ENV_SHORT=$(env_short "$ENV_NAME")
TOTAL=$(( ${#ENCODERS[@]} * $(echo $SEEDS | wc -w) ))
COUNT=0

for ENC in "${ENCODERS[@]}"; do
    TRAIN_ARGS=$(get_train_args "$ENC")
    for SEED in $SEEDS; do
        COUNT=$((COUNT + 1))
        RUN_NAME="sweep_${ENV_SHORT}_${ENC}_s${SEED}"
        LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  [$COUNT/$TOTAL] Enc=$ENC  Seed=$SEED  Env=$ENV_SHORT  Run=$RUN_NAME"
        echo "  $(date)"
        echo "════════════════════════════════════════════════════════════════"

        # Skip if already in CSV
        if grep -q "^${ENC},${SEED},${ENV_SHORT}," "$RESULTS_CSV" 2>/dev/null; then
            echo "  → already in results CSV, skipping"
            continue
        fi

        cd "$PROJECT_ROOT/discrete_mbrl/model_free"
        export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
        export TORCHDYNAMO_DISABLE=1
        export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

        # Continuous encoder path (VAE) needs wm_aux disabled — the PPO aux loss
        # uses cross-entropy against discrete code indices, which isn't defined
        # for continuous latents. WM still trains online via trans_trainer.
        if [[ "$ENC" == "vae" ]]; then
            WM_AUX_COEF=0.0
        else
            WM_AUX_COEF=0.1
        fi

        echo "  Training $MF_STEPS steps... (log: $LOG_FILE)  wm_aux_coef=$WM_AUX_COEF"

        "$PYTHON_BIN" -u train.py \
            --env_name "$ENV_NAME" \
            $TRAIN_ARGS \
            --seed "$SEED" \
            --mf_steps $MF_STEPS \
            --batch_size $BATCH_SIZE \
            --num_envs $NUM_ENVS \
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
            --run_name "$RUN_NAME" \
            --device "$DEVICE" \
            --save \
            --use_world_model \
            --wm_aux_coef $WM_AUX_COEF \
            --wm_train_freq 1 \
            --trans_model_version 1 \
            --trans_hidden 256 \
            --trans_depth 3 \
            --use_semantic_aux \
            --sem_aux_coef 0.05 \
            --sem_head_hidden 128 \
            --sem_head_version 2 \
            --sem_n_classes 11 \
            --sem_aux_start_reward 0.0 \
            --sem_class_weights \
            --sem_class_weight_power 1.0 \
            --sem_focal_gamma 2.0 \
            --sem_pre_vq \
            > "$LOG_FILE" 2>&1

        TRAIN_EXIT=$?
        if [[ $TRAIN_EXIT -ne 0 ]]; then
            echo "  ✗ Training failed (exit $TRAIN_EXIT). See $LOG_FILE"
            continue
        fi

        BEST_REWARD=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
        FINAL_REWARD=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
        OVERALL_AVG=$(grep -aoP 'Overall average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
        echo "  ✓ Training done: best=$BEST_REWARD final=$FINAL_REWARD avg=$OVERALL_AVG"

        # ── Multi-step WM analysis ──
        BEST_MODEL="$PROJECT_ROOT/discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"
        OUT_JSON="$WM_OUT_DIR/${RUN_NAME}.json"

        if [[ -f "$BEST_MODEL" ]]; then
            echo "  Running multi-step WM analysis..."
            cd "$PROJECT_ROOT/discrete_mbrl"
            "$PYTHON_BIN" -u analyze_wm_multistep.py \
                --model_path "$BEST_MODEL" \
                --n_rollouts 500 \
                --probe_frames 20000 \
                --horizons 1 3 5 10 \
                --device "$DEVICE" \
                --output_json "$OUT_JSON" \
                > "$LOG_DIR/${RUN_NAME}_analysis.log" 2>&1 || true

            if [[ -f "$OUT_JSON" ]]; then
                # Extract summary fields into CSV
                "$PYTHON_BIN" -c "
import json
d = json.load(open('$OUT_JSON'))
def g(pc, cls): return pc.get(str(cls), {}).get('acc', None)
pa = d.get('probe_acc_per_class', {})
wm = d.get('wm_acc_per_class_per_horizon', {})
row = ['$ENC','$SEED','$ENV_SHORT','$BEST_REWARD','$FINAL_REWARD','$OVERALL_AVG']
for cls in [2, 4, 5, 8, 10]:
    v = g(pa, cls); row.append(f'{v:.4f}' if v is not None else 'NA')
for k in [1, 10]:
    for cls in [2, 4, 5, 8, 10]:
        v = g(wm.get(str(k), {}), cls); row.append(f'{v:.4f}' if v is not None else 'NA')
for k in [1, 5, 10]:
    r = d.get('pearson_r_per_horizon', {}).get(str(k))
    row.append(f'{r:+.3f}' if r is not None else 'NA')
print(','.join(row))
" >> "$RESULTS_CSV"
                echo "  → Saved to $RESULTS_CSV"
            else
                echo "  ⚠ Analysis failed, no JSON output"
            fi
        else
            echo "  ⚠ No best model at $BEST_MODEL, skipping analysis"
        fi
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SWEEP COMPLETE — Results in $RESULTS_CSV"
echo "════════════════════════════════════════════════════════════════"
cat "$RESULTS_CSV"
