#!/bin/bash
# Score every checkpoint of a run against the held-out episodes.
#
# The trainer has no validation path (DatasetFactory.build asserts
# eval_strategy == "no"), so "is it still learning or now memorising?" can only
# be answered after the fact, by replaying held-out episodes through each
# checkpoint. That is what this does.
#
# Read the numbers with the run-to-run spread in mind: flow-matching sampling is
# not seeded, so the same checkpoint scores within about ±15% across repeats
# (RLDX-1-FFW-SH5.md appendix A.4). Only treat a gap larger than that as real.
#
# Usage:
#   bash run_scripts/eval/eval_holdout_all.sh                    # defaults below
#   STEPS="10000 20000" EVAL_STEPS=128 bash .../eval_holdout_all.sh   # EVAL_STEPS: frames per episode (default 1000 = full)

set -uo pipefail

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"

CKPT_NAME="${CKPT_NAME:-rldx1_ft_ffw_sh5_0826}"
RUN_DIR="${RUN_DIR:-$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME/$CKPT_NAME}"
DATASET="${DATASET:-$BASE_DIR/dataset/rlwrld_demo_0826_holdout}"
ACTION_HORIZON="${ACTION_HORIZON:-40}"
TRAJ_IDS="${TRAJ_IDS:-0 1 2 3 4 5 6 7 8}"   # holdout episodes are renumbered 0..8
STEPS="${STEPS:-}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/outputs/eval}"

if [[ -z "$STEPS" ]]; then
    STEPS=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null \
            | sed 's/.*checkpoint-//' | sort -n | tr '\n' ' ')
fi
[[ -z "${STEPS// }" ]] && { echo "ERROR: no checkpoints under $RUN_DIR" >&2; exit 1; }

if [[ -z "${RUN_PREFIX:-}" ]]; then
    if [[ -d "$BASE_DIR/.pixi/envs/rldx-sm100" ]]; then
        RUN_PREFIX="pixi run -e rldx-sm100"
    else
        RUN_PREFIX="uv run"
    fi
fi
command -v pixi >/dev/null 2>&1 || export PATH="/NHNHOME/doyoung/.pixi/bin:$PATH"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/holdout_summary.txt"
: > "$SUMMARY"

echo "[i] run       : $RUN_DIR"
echo "[i] holdout   : $DATASET"
echo "[i] horizon   : $ACTION_HORIZON   traj: $TRAJ_IDS"
echo "[i] checkpoints: $STEPS"
echo

for STEP in $STEPS; do
    CKPT="$RUN_DIR/checkpoint-$STEP"
    [[ -d "$CKPT" ]] || { echo "[!] missing $CKPT — skip"; continue; }
    LOG="$OUT_DIR/holdout_${STEP}.log"
    echo "=== checkpoint-$STEP ==="
    $RUN_PREFIX python run_scripts/eval/open_loop_eval_verbose.py \
        --model-path "$CKPT" \
        --dataset-path "$DATASET" \
        --embodiment-tag GENERAL_EMBODIMENT \
        --traj-ids $TRAJ_IDS \
        --action-horizon "$ACTION_HORIZON" \
        --steps "${EVAL_STEPS:-1000}" \
        --save-plot-path "$OUT_DIR/holdout_${STEP}.png" > "$LOG" 2>&1
    # NOTE: do not read the log's "Average MSE across all trajs" line — it is
    # printed once per trajectory and each print covers only what that call has
    # seen, so the last one is the final trajectory's score, not the mean.
    # Average the per-trajectory lines instead.
    read -r MSE MAE < <(grep -oE "MSE for trajectory [0-9]+: [0-9.]+, MAE: [0-9.]+" "$LOG" \
        | awk -F'[:,]' '{m+=$2; a+=$4; n++} END {if(n) printf "%.6f %.6f", m/n, a/n}')
    printf "checkpoint-%-6s  MSE %-12s  MAE %-12s\n" "$STEP" "${MSE:-FAILED}" "${MAE:-FAILED}" \
        | tee -a "$SUMMARY"
done

echo
echo "===== held-out summary ====="
cat "$SUMMARY"
echo "(same checkpoint varies ~±15% between runs — ignore gaps smaller than that)"
