#!/usr/bin/env bash
# Full-rigor sweep chaining (A) DK16 resolution + (B) DK8 multi-seed.
# Runs sequentially on one GPU; each training is followed by probe + codebook.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="$PROJECT_ROOT/logs/full_rigor"

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.mplconfig"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

RESULTS_CSV="$LOG_DIR/full_rigor_results.csv"
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "group,variant,seed,best_reward,final_reward,probe_overall,probe_door,probe_key,probe_goal,cb_dead,cb_active" > "$RESULTS_CSV"
fi

post_analyze() {  # args: run_name env_name variant seed group
    local run_name="$1" env="$2" variant="$3" seed="$4" group="$5"
    local model_dir="$PROJECT_ROOT/discrete_mbrl/model_free/models/${env}"
    local model="$model_dir/${run_name}_best_model.pt"
    local tlog="$LOG_DIR/${run_name}_train.log"
    local plog="$LOG_DIR/${run_name}_probe.log"
    local cbjson="$LOG_DIR/${run_name}_codebook.json"
    local cblog="$LOG_DIR/${run_name}_codebook.log"

    local best final
    best=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$tlog" | tail -1 || echo NA)
    final=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$tlog" | tail -1 || echo NA)

    if [[ -f "$model" ]]; then
        cd "$PROJECT_ROOT/discrete_mbrl"
        "$PYTHON_BIN" -u probe_semantics.py \
            --model_path "$model" --n_frames 30000 --probe_epochs 20 \
            --device "$DEVICE" > "$plog" 2>&1 || true
        "$PYTHON_BIN" -u analyze_codebook_usage.py \
            --model_path "$model" --n_frames 8000 --device "$DEVICE" \
            --output_json "$cbjson" > "$cblog" 2>&1 || true
    fi

    local po pd pk pg cbd cba
    po=$(grep -a "Overall per-position accuracy" "$plog" 2>/dev/null | grep -oP '[\d.]+(?=%)' | head -1 || echo NA)
    pd=$(grep -a "door"  "$plog" 2>/dev/null | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo NA)
    pk=$(grep -a " key " "$plog" 2>/dev/null | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo NA)
    pg=$(grep -a "goal"  "$plog" 2>/dev/null | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo NA)
    if [[ -f "$cbjson" ]]; then
        cbd=$("$PYTHON_BIN" -c "import json;d=json.load(open('$cbjson'));print(d.get('n_dead','NA'))")
        cba=$("$PYTHON_BIN" -c "import json;d=json.load(open('$cbjson'));print(d.get('n_active','NA'))")
    else cbd=NA; cba=NA; fi

    echo "${group},${variant},${seed},${best},${final},${po},${pd},${pk},${pg},${cbd},${cba}" >> "$RESULTS_CSV"
    echo "  → $variant s$seed  best=$best  final=$final  probe=${po}%  door=${pd}%  key=${pk}%  goal=${pg}%  dead=${cbd}"
}

# ──────────── (A) DK-16 resolution ────────────
ENV16="MiniGrid-DoorKey-16x16-v0"

run_dk16() {  # args: run_name cb thr
    local run_name="$1" cb="$2" thr="$3"
    local tlog="$LOG_DIR/${run_name}_train.log"
    echo ""
    echo "════ DK16 $run_name  cb=$cb thr=$thr ════"
    if grep -q ",${run_name}," "$RESULTS_CSV"; then echo "  already done"; return; fi
    cd "$PROJECT_ROOT/discrete_mbrl/model_free"
    "$PYTHON_BIN" -u train.py \
        --env_name "$ENV16" \
        --ae_model_type vqvae --ae_model_version 6 \
        --codebook_size "$cb" --embedding_dim 64 --filter_size 16 \
        --dead_code_threshold "$thr" --seed 1 \
        --mf_steps 8000000 --batch_size 4096 --num_envs 16 \
        --ppo_iters 10 --ppo_batch_size 64 --ppo_entropy_coef 0.01 \
        --ppo_gae_lambda 0.95 --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
        --learning_rate 1e-4 --e2e_loss \
        --encoder_lr 1e-5 --encoder_lr_cosine \
        --encoder_snapback --snapback_threshold 0.5 --snapback_patience 100 --snapback_min_reward 0.05 \
        --ortho_init --model_dir .. --run_name "$run_name" --device "$DEVICE" --save \
        --use_world_model --wm_aux_coef 0.1 --wm_train_freq 1 \
        --trans_model_type discrete --trans_model_version 1 --trans_hidden 256 --trans_depth 3 \
        --use_semantic_aux --sem_aux_coef 0.08 --sem_head_hidden 128 --sem_head_version 2 \
        --sem_n_classes 11 --sem_aux_start_reward 0.0 --sem_class_weights \
        --sem_focal_gamma 0.0 --sem_pre_vq \
        > "$tlog" 2>&1 || echo "  ✗ train failed"
    post_analyze "$run_name" "$ENV16" "cb${cb}_thr${thr}" 1 "A_dk16"
}

run_dk16 "rigor_dk16_cb512_thr2"  512 2.0
run_dk16 "rigor_dk16_cb256_thr1"  256 1.0

# ──────────── (B) DK-8 multi-seed ────────────
ENV8="MiniGrid-DoorKey-8x8-v0"

run_dk8() {  # args: variant seed   — variant ∈ {v2,v5dc,v6,vae}
    local variant="$1" seed="$2"
    local run_name="rigor_dk8_${variant}_s${seed}"
    local tlog="$LOG_DIR/${run_name}_train.log"
    echo ""
    echo "════ DK8 $variant seed=$seed ════"
    if grep -q ",${run_name}," "$RESULTS_CSV"; then echo "  already done"; return; fi
    cd "$PROJECT_ROOT/discrete_mbrl/model_free"

    local args=(
        --env_name "$ENV8" --embedding_dim 64 --filter_size 8 --seed "$seed"
        --mf_steps 5000000 --batch_size 4096 --num_envs 16
        --ppo_iters 10 --ppo_batch_size 64 --ppo_entropy_coef 0.01
        --ppo_gae_lambda 0.95 --ppo_norm_advantages --ppo_max_grad_norm 0.5
        --learning_rate 1e-4 --e2e_loss
        --encoder_lr 1e-5 --encoder_lr_cosine
        --encoder_snapback --snapback_threshold 0.5 --snapback_patience 100 --snapback_min_reward 0.0
        --ortho_init --model_dir .. --run_name "$run_name" --device "$DEVICE" --save
        --use_semantic_aux --sem_aux_coef 0.05 --sem_head_hidden 128 --sem_head_version 2
        --sem_n_classes 11 --sem_aux_start_reward 0.0 --sem_class_weights
        --sem_focal_gamma 2.0 --sem_pre_vq
    )

    case "$variant" in
        v2)
            args+=( --ae_model_type vqvae --ae_model_version 2 --codebook_size 64
                    --use_world_model --wm_aux_coef 0.1 --wm_train_freq 1
                    --trans_model_type discrete --trans_model_version 1
                    --trans_hidden 256 --trans_depth 3 ) ;;
        v5dc)
            args+=( --ae_model_type vqvae --ae_model_version 5 --codebook_size 64
                    --dead_code_threshold 2.0
                    --use_world_model --wm_aux_coef 0.1 --wm_train_freq 1
                    --trans_model_type discrete --trans_model_version 1
                    --trans_hidden 256 --trans_depth 3 ) ;;
        v6)
            args+=( --ae_model_type vqvae --ae_model_version 6 --codebook_size 256
                    --dead_code_threshold 1.0
                    --use_world_model --wm_aux_coef 0.1 --wm_train_freq 1
                    --trans_model_type discrete --trans_model_version 1
                    --trans_hidden 256 --trans_depth 3 ) ;;
        vae)
            args+=( --ae_model_type vae --ae_model_version 6 ) ;;
    esac

    "$PYTHON_BIN" -u train.py "${args[@]}" > "$tlog" 2>&1 || echo "  ✗ train failed"
    post_analyze "$run_name" "$ENV8" "$variant" "$seed" "B_dk8"
}

for SEED in 1 2; do
    for VAR in v2 v5dc v6 vae; do
        run_dk8 "$VAR" "$SEED"
    done
done

echo ""
echo "════════════════════════════════════════════"
echo "  FULL-RIGOR SWEEP COMPLETE"
echo "════════════════════════════════════════════"
cat "$RESULTS_CSV"
