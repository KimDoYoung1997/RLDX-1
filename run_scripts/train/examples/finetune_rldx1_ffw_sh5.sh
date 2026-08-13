#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on FFW-SH5 (arms + hands)"
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=slurm_out/%j-rldx1_ft_ffw_sh5.out
#SBATCH --error=slurm_out/%j-rldx1_ft_ffw_sh5.err

set -euo pipefail

# Fine-tune RLDX-1-PT on the FFW-SH5 rev1 teleop dataset
# (HF: learner1119/ffw_sh5_0812, LeRobot v2.1, 7 episodes / 1253 frames @ 10fps).
#
# Modalities (see rldx/configs/data/ffw_sh5_rev1_config_wholebody.py):
#   video  : cam_head only (wrist cams present in the data but unused)
#   state  : left_arm(7) right_arm(7) left_hand(20) right_hand(20)  -> 54 of 57 dims
#   action : same 4 groups, absolute joint targets, 16-step horizon
#   head_joint1/2 (54:56) and lift_joint (56) are intentionally dropped.
#
# Prerequisites (one-time, see CLAUDE.md):
#   hf download learner1119/ffw_sh5_0812 --repo-type dataset --local-dir dataset/ffw_sh5_0812
#   dataset/ffw_sh5_0812/meta/modality.json  written by hand
#   dataset/ffw_sh5_0812/meta/stats.json     generated via rldx/data/stats.py

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_ffw_sh5_wholebody"
RUN_NAME="$CKPT_NAME"

NUM_GPUS=4
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
EMBODIMENT_TAG="GENERAL_EMBODIMENT"
DATASET_PATH="${DATASET_PATH:-$BASE_DIR/dataset/ffw_sh5_0812}"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/ffw_sh5_rev1_config_wholebody.py"
CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"

# Run sizing. SMOKE=1 shrinks the run to ~20 steps just to prove the pipeline
# assembles, the loss is finite, and a checkpoint gets written.
if [[ "${SMOKE:-0}" == "1" ]]; then
    MAX_STEPS=20; SAVE_STEPS=20; GLOBAL_BATCH_SIZE=8; SAVE_TOTAL_LIMIT=1
else
    # 1253 frames total — keep the run short and checkpoint often.
    MAX_STEPS=10000; SAVE_STEPS=5000; GLOBAL_BATCH_SIZE=64; SAVE_TOTAL_LIMIT=5
fi

ARGS=(
    --n-cog-tokens 64
    # RLDX-1-PT is the video-input checkpoint -> video-length 4.
    # Switch to RLWRLD/RLDX-1-PT-IMG together with --video-length 1.
    --video-length 4

    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --dataloader-num-workers 4

    --base-model-path "$BASE_MODEL_PATH"
    --output-dir "$CKPT_DIR"
    --num-gpus "$NUM_GPUS"

    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --learning-rate 1e-4

    # TODO: tactile (observation.tactile, 90-dim) as a physics modality.
    #   1) add to meta/modality.json:
    #        "tactile": {"left_hand":  {"start":  0, "end": 45},
    #                    "right_hand": {"start": 45, "end": 90}}
    #   2) add a matching "tactile" ModalityConfig to the config file
    #   3) uncomment:
    # --use-physics --physics-keys tactile --physics-dims 90 --new-param-warmup-steps 500

    --experiment-name "$RUN_NAME"
)

# wandb needs `wandb login` first. Run with USE_WANDB=0 to skip it entirely.
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
fi

cd "$BASE_DIR"
mkdir -p slurm_out
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

# Pick the launcher. The uv/.venv torch is a cu126 wheel whose CUBINs stop at
# sm_90, so on a datacenter-Blackwell box (B200, sm_100) it dies at the first
# CUDA launch. The `rldx-sm100` pixi env only exists where it was built (see the
# ffw-sh5-b200 branch), so its presence is the signal:
#
#   .pixi/envs/rldx-sm100 present -> pixi run -e rldx-sm100
#   otherwise                     -> uv run   (upstream default, sm_80/sm_90)
#
# Force either with RUN_PREFIX="uv run" / RUN_PREFIX="pixi run -e rldx-sm100".
if [[ -z "${RUN_PREFIX:-}" ]]; then
    if [[ -d "$BASE_DIR/.pixi/envs/rldx-sm100" ]]; then
        RUN_PREFIX="pixi run -e rldx-sm100"
    else
        RUN_PREFIX="uv run"
    fi
fi

# pixi lives on the persistent volume (see /NHNHOME/doyoung/setup_rldx.sh), which
# is not on PATH in non-interactive shells — sbatch, cron and `bash script.sh`
# all skip ~/.bashrc. Resolve it so the script does not depend on the caller.
if [[ "$RUN_PREFIX" == pixi* ]] && ! command -v pixi >/dev/null 2>&1; then
    for _p in "${PIXI_HOME:-}/bin" /NHNHOME/doyoung/.pixi/bin "$HOME/.pixi/bin"; do
        if [[ -x "$_p/pixi" ]]; then export PATH="$_p:$PATH"; break; fi
    done
    if ! command -v pixi >/dev/null 2>&1; then
        echo "ERROR: the rldx-sm100 env exists but the pixi binary is not on PATH." >&2
        echo "       Run 'bash /NHNHOME/doyoung/setup_rldx.sh all' first." >&2
        exit 1
    fi
fi
echo "[i] launcher: $RUN_PREFIX"
$RUN_PREFIX torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py "${ARGS[@]}"
