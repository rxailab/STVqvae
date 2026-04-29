#!/usr/bin/env bash
# Master queue: runs W1 (DK-8 seeds 4,5 + DK-16 seeds 1-3) + W2 (VAE+WM) serially.
# Expected wall-clock: ~40 hours on one 4090.
#
# Usage:
#   bash run_paper_sweep.sh                  # run everything
#   bash run_paper_sweep.sh w1_dk8           # run one workstream
#
# Safe to interrupt and resume — each sweep skips runs already in its CSV.
set -euo pipefail

WORKTREE="/home/xiar3/experiments/STVqvae/.claude/worktrees/sharp-buck-fef000"
MAIN_TREE="/home/xiar3/experiments/STVqvae"
QUEUE_LOG="$MAIN_TREE/logs/run_paper_sweep.log"
mkdir -p "$MAIN_TREE/logs"

STAGES="${1:-w1_dk8 w2_vae w1_dk16_v6 w1_dk16_v5dc}"

run_stage() {
    local name="$1"; shift
    echo ""
    echo "########################################################################"
    echo "# STAGE: $name  at $(date)"
    echo "########################################################################"
    "$@"
    echo "# STAGE $name DONE at $(date)"
}

for STAGE in $STAGES; do
    case "$STAGE" in
        w1_dk8)
            # Seeds 4, 5 on DK-8 for v2, v5dc, v6 — 6 runs × ~2h = 12h
            run_stage "W1_DK8" env SEEDS="4 5" ENV_NAME="MiniGrid-DoorKey-8x8-v0" \
                bash "$WORKTREE/sweep_multiseed_v2.sh" v2 v5dc v6
            ;;
        w2_vae)
            # VAE + continuous MLP WM × 5 seeds on DK-8 — 5 runs × ~2h = 10h
            run_stage "W2_VAE_WM" env SEEDS="1 2 3 4 5" ENV_NAME="MiniGrid-DoorKey-8x8-v0" \
                bash "$WORKTREE/sweep_multiseed_v2.sh" vae
            ;;
        w1_dk16_v6)
            # DK-16 seeds 1,2,3 on v6 — 3 runs × ~3h = 9h
            run_stage "W1_DK16_v6" env SEEDS="1 2 3" \
                bash "$WORKTREE/sweep_doorkey16_v2.sh" v6
            ;;
        w1_dk16_v5dc)
            # DK-16 seeds 1,2,3 on v5dc — 3 runs × ~3h = 9h
            run_stage "W1_DK16_v5dc" env SEEDS="1 2 3" \
                bash "$WORKTREE/sweep_doorkey16_v2.sh" v5dc
            ;;
        *)
            echo "Unknown stage: $STAGE" >&2; exit 1
            ;;
    esac
done 2>&1 | tee -a "$QUEUE_LOG"

echo ""
echo "########################################################################"
echo "# ALL STAGES COMPLETE at $(date)"
echo "########################################################################"
