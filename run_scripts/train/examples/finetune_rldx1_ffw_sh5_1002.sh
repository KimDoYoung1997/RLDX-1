#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on FFW-SH5 1002 merge (26-dim action, chunk 40, 20 fps)"
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=slurm_out/%j-rldx1_ft_ffw_sh5_1002.out
#SBATCH --error=slurm_out/%j-rldx1_ft_ffw_sh5_1002.err

set -euo pipefail

# Fine-tune RLDX-1-PT on the 1002 merge (rlwrld_demo_merged_1002).
# Same recipe as finetune_rldx1_ffw_sh5_0826.sh — only the data changed, so
# the 0826 reasoning in RLDX-1-FFW-SH5.md still applies.
#
#   data      dataset/rlwrld_demo_merged_1002_train   166 ep / 85,494 frames
#             (held-out: dataset/rlwrld_demo_merged_1002_holdout, merged ep
#             61/66/71/76 = one per box_pose_case 1-4; HF has both splits)
#   action    26-dim, chunk 40 — the same 26 commanded dims as 0826 (verified)
#   image     --image-max-area 98304 -> 224x384, 84 vision tokens
#   modules   motion / memory / physics all OFF — RLDX-1-PT has none of them
#
# This data is 20 fps, not 10: chunk 40 now predicts 2.0 s (not 4.0 s), and
# --video-length 4 at stride 2 spans 0.3 s. Deployment: execute the first 20
# (= 1.0 s) and re-query. The execution length is a client-side choice and is
# not baked into the checkpoint.

export NO_ALBUMENTATIONS_UPDATE=1
# Base checkpoint cache on the shared volume, not the per-session home dir.
export HF_HOME="${HF_HOME:-/NHNHOME/doyoung/.cache/huggingface}"

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"

# ---------------------------------------------------------------------------
# Knobs. Override from the environment, e.g.
#   SAVE_STEPS=5000 MAX_STEPS=20000 bash run_scripts/train/.../finetune_...sh
# ---------------------------------------------------------------------------
BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
DATASET_PATH="${DATASET_PATH:-$BASE_DIR/dataset/rlwrld_demo_merged_1002_train}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-$BASE_DIR/rldx/configs/data/ffw_sh5_rev1_config_used26_h40.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-GENERAL_EMBODIMENT}"

NUM_GPUS="${NUM_GPUS:-4}"
ACTION_HORIZON="${ACTION_HORIZON:-40}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
IMAGE_MAX_AREA="${IMAGE_MAX_AREA:-98304}"
STATE_DROPOUT="${STATE_DROPOUT:-0.3}"

MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"

CKPT_NAME="${CKPT_NAME:-rldx1_ft_ffw_sh5_1002}"

# SMOKE=1 proves the pipeline assembles without touching the real run's output
# directory — experiment.py always resumes, so a leftover smoke checkpoint in
# the real directory would silently inherit its step counter.
if [[ "${SMOKE:-0}" == "1" ]]; then
    MAX_STEPS=20; SAVE_STEPS=20; SAVE_TOTAL_LIMIT=1; GLOBAL_BATCH_SIZE=16
    CKPT_NAME="${CKPT_NAME}_smoke"
fi

RUN_NAME="$CKPT_NAME"
CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"

# Resuming is deliberate (experiment.py passes resume_from_checkpoint=True) but
# should never be a surprise (RLDX-1-FFW-SH5.md 부록 A.3).
if compgen -G "$CKPT_DIR/$RUN_NAME/checkpoint-*" >/dev/null; then
    _last=$(ls -d "$CKPT_DIR/$RUN_NAME"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tail -1)
    echo "[!] $CKPT_DIR/$RUN_NAME already holds checkpoints (latest: $_last)."
    echo "[!] This run will RESUME from it, not start fresh."
    echo "[!] For a fresh run: CKPT_NAME=<something-new> bash $0"
fi

# One checkpoint is ~61 GB (8B weights + optimizer state), so say up front what
# this run will occupy rather than discovering it at 90% disk.
_n_saves=$(( MAX_STEPS / SAVE_STEPS ))
_kept=$(( _n_saves < SAVE_TOTAL_LIMIT ? _n_saves : SAVE_TOTAL_LIMIT ))
echo "[i] saves at every $SAVE_STEPS steps -> $_n_saves total, keeping $_kept (~$(( _kept * 61 )) GB)"
if (( _n_saves > SAVE_TOTAL_LIMIT )); then
    echo "[!] SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT < $_n_saves saves: the EARLIEST checkpoints get deleted."
    echo "[!] Held-out MSE often bottoms out early, so evaluate before they roll off."
fi

ARGS=(
    --base-model-path "$BASE_MODEL_PATH"
    --video-length 4                      # {-6,-4,-2,0} with the default stride 2 (§1.1)
    --n-cog-tokens 64
    --action-horizon "$ACTION_HORIZON"    # must equal len(delta_indices) — assembly.py:304

    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --dataloader-num-workers "$DATALOADER_WORKERS"

    --output-dir "$CKPT_DIR"
    --experiment-name "$RUN_NAME"
    --num-gpus "$NUM_GPUS"

    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --lr-scheduler-type cosine
    --warmup-ratio 0.05
    --weight-decay 1e-5
    --max-grad-norm 1.0

    # Freeze the vision encoder and the LLM except its top 4 layers. Fully
    # freezing the backbone costs 26 points in the paper's ablation (Table 6).
    --tune-top-llm-layers 4
    # PT's config.json says 0.0, but assembly.py:166 lets the CLI win, so this
    # is what actually applies. Paper uses 0.3 for OpenArm and FR3.
    --state-dropout-prob "$STATE_DROPOUT"
    --image-max-area "$IMAGE_MAX_AREA"

    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
)

# wandb is off by default: no key is present in $KDY_ROOT/.secrets. Set
# USE_WANDB=1 after creating one to get loss curves.
if [[ "${USE_WANDB:-0}" == "1" ]]; then
    ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT:-rldx-finetune}")
fi

# Launcher: the uv/.venv torch is cu126 (CUBINs stop at sm_90) and dies on a
# B200. Presence of the pixi env is the signal (RLDX-1-FFW-SH5.md 부록 A.6).
if [[ -z "${RUN_PREFIX:-}" ]]; then
    if [[ -d "$BASE_DIR/.pixi/envs/rldx-sm100" ]]; then
        RUN_PREFIX="pixi run -e rldx-sm100"
    else
        RUN_PREFIX="uv run"
    fi
fi
if [[ "$RUN_PREFIX" == pixi* ]] && ! command -v pixi >/dev/null 2>&1; then
    for _p in "${PIXI_HOME:-}/bin" /NHNHOME/doyoung/.pixi/bin "$HOME/.pixi/bin"; do
        [[ -x "$_p/pixi" ]] && { export PATH="$_p:$PATH"; break; }
    done
    command -v pixi >/dev/null 2>&1 || {
        echo "ERROR: rldx-sm100 env exists but pixi is not on PATH." >&2
        echo "       Run 'bash /NHNHOME/doyoung/setup_rldx.sh all' first." >&2
        exit 1
    }
fi

mkdir -p slurm_out
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}"

echo "[i] launcher   : $RUN_PREFIX"
echo "[i] dataset    : $DATASET_PATH"
echo "[i] modality   : $MODALITY_CONFIG_PATH  (action_horizon=$ACTION_HORIZON)"
echo "[i] output     : $CKPT_DIR/$RUN_NAME"
echo "[i] batch/steps: $GLOBAL_BATCH_SIZE x $MAX_STEPS"

$RUN_PREFIX torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    rldx/experiment/launch_train.py "${ARGS[@]}"
