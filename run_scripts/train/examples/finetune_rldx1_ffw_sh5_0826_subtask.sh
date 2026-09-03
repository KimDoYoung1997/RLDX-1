#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on FFW-SH5 0826 merge, subtask-conditioned language"
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=slurm_out/%j-rldx1_ft_ffw_sh5_0826_subtask.out
#SBATCH --error=slurm_out/%j-rldx1_ft_ffw_sh5_0826_subtask.err

set -euo pipefail

# Fine-tune RLDX-1-PT on the merged FFW-SH5 teleop data, with SUBTASK-level
# language conditioning. Identical to finetune_rldx1_ffw_sh5_0826.sh except
# for the dataset and the run name: the _subtask dataset copy stamps a
# per-frame task_index (1=bring / 2=flip / 3=push, from human segment labels
# in meta/subtask_labels.jsonl), and the loader already resolves task_index
# per frame (lerobot_episode_loader.py:350) — so this is a pure data change,
# zero training-code change. Hyperparameters match the baseline run exactly
# so the comparison isolates the conditioning.
#
# Held-out scoring must use dataset/rlwrld_demo_0826_holdout_subtask (same 9
# episodes, subtask-conditioned = oracle sequencer). Deployment executes 20
# of the 40-step chunk (2.0 s) and re-queries, so a boundary-crossing chunk
# is mostly discarded in practice.
#
# Everything here is argued for in RLDX-1-FFW-SH5.md; the section
# references below point at the reasoning rather than repeating it.
#
#   data      dataset/rlwrld_demo_0826_train   72 ep / 29,661 frames (§5.3)
#             (held-out 9 ep live in dataset/rlwrld_demo_0826_holdout)
#   action    26-dim, chunk 40                 (§12 Q2, §13.7)
#   image     --image-max-area 98304 -> 224x384, 84 vision tokens (§5.2)
#   modules   motion / memory / physics all OFF — RLDX-1-PT has none of them (§2)
#
# Deployment note: chunk 40 is what the policy predicts; execute only the first
# 20 (= 2.0 s at 10 fps) and re-query. That ratio matches the paper's ALLEX
# setup and is a *client-side* choice — it is not baked into the checkpoint
# (§13.5, §13.7).

export NO_ALBUMENTATIONS_UPDATE=1

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"

# ---------------------------------------------------------------------------
# Knobs. Override from the environment, e.g.
#   SAVE_STEPS=5000 MAX_STEPS=20000 bash run_scripts/train/.../finetune_...sh
# ---------------------------------------------------------------------------
BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
DATASET_PATH="${DATASET_PATH:-$BASE_DIR/dataset/rlwrld_demo_0826_train_subtask}"
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

CKPT_NAME="${CKPT_NAME:-rldx1_ft_ffw_sh5_0826_subtask}"

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
