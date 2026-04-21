#!/bin/bash
# Run semantic probes on all 12 completed sweep models and update CSV
set -e
PROJECT_ROOT="/home/xiar3/experiments/STVqvae"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
LOG_DIR="$PROJECT_ROOT/logs/sweep"
RESULTS_CSV="$LOG_DIR/sweep_results.csv"
ENV_NAME="MiniGrid-DoorKey-8x8-v0"
DEVICE="cuda"

cd "$PROJECT_ROOT/discrete_mbrl"

ENCODERS=(v2 v5 v6 v9)
SEEDS=(1 2 3)

# Rewrite CSV header
echo "encoder,seed,best_reward,final_reward,overall_avg,probe_overall,probe_wall,probe_floor,probe_door,probe_key,probe_goal,probe_agent,codebook_util" > "$RESULTS_CSV"

for ENC in "${ENCODERS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="sweep_doorkey8_${ENC}_s${SEED}"
        BEST_MODEL="$PROJECT_ROOT/discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"
        PROBE_LOG="$LOG_DIR/${RUN_NAME}_probe.log"
        TRAIN_LOG="$LOG_DIR/${RUN_NAME}.log"

        # Extract training metrics from log
        # tqdm mixes reward lines with progress bars via \r; use specific keyword lookbehind
        BEST_REWARD=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$TRAIN_LOG" | tail -1 || echo "NA")
        FINAL_REWARD=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$TRAIN_LOG" | tail -1 || echo "NA")
        OVERALL_AVG=$(grep -aoP 'Overall average reward:\s+\K[0-9.]+' "$TRAIN_LOG" | tail -1 || echo "NA")

        echo "=== Probing ${ENC} s${SEED} (best=${BEST_REWARD}) ==="

        if [[ -f "$BEST_MODEL" ]]; then
            TORCHDYNAMO_DISABLE=1 "$PYTHON_BIN" -u probe_semantics.py \
                --model_path "$BEST_MODEL" \
                --n_frames 50000 \
                --probe_epochs 20 \
                --device "$DEVICE" \
                > "$PROBE_LOG" 2>&1

            PROBE_OVERALL=$(grep -a "Overall per-position accuracy" "$PROBE_LOG" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_WALL=$(grep   -a "wall"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_FLOOR=$(grep  -a "empty" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_DOOR=$(grep   -a "door"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_KEY=$(grep    -a "key"   "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_GOAL=$(grep   -a "goal"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_AGENT=$(grep  -a "agent" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            echo "  ✓ overall=${PROBE_OVERALL}%  goal=${PROBE_GOAL}%"
        else
            echo "  ⚠ Model not found: $BEST_MODEL"
            PROBE_OVERALL="NA"; PROBE_WALL="NA"; PROBE_FLOOR="NA"
            PROBE_DOOR="NA"; PROBE_KEY="NA"; PROBE_GOAL="NA"; PROBE_AGENT="NA"
        fi

        echo "${ENC},${SEED},${BEST_REWARD},${FINAL_REWARD},${OVERALL_AVG},${PROBE_OVERALL},${PROBE_WALL},${PROBE_FLOOR},${PROBE_DOOR},${PROBE_KEY},${PROBE_GOAL},${PROBE_AGENT},NA" >> "$RESULTS_CSV"
    done
done

echo ""
echo "=== All probes done. Results in $RESULTS_CSV ==="
cat "$RESULTS_CSV"
