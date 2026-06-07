#!/bin/bash
# ── SLURM resource request ────────────────────────────────────────────────────
#SBATCH -J gsm8k_grpo
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -p mit_normal_gpu
#SBATCH --array=0-15%2                 # 16 tasks (0–15), at most 2 running at once
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%A_%a.out

# Phase 4 — GRPO training (16 runs via SLURM array, up to 2 concurrent).
# Array index 0-15 maps to all 16 (sft_sel, grpo_sel) combinations.
# To smoke-test: DRY_RUN=1 sbatch submit_grpo.sh
#
# Environment variables (set by Makefile or manually):
#   MODEL       full HuggingFace model ID  (default: Qwen/Qwen2.5-0.5B-Instruct)
#   MODEL_NAME  short name used in paths   (default: basename of MODEL)
#   SEED        random seed                (default: 42)
#   DRY_RUN     set to 1 for a 20-step smoke test

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img

MODEL=${MODEL:-"Qwen/Qwen2.5-0.5B-Instruct"}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-""}

CKPT_DIR=$REPO_DIR/checkpoints/$MODEL_NAME
LOGS_DIR=$REPO_DIR/logs/$MODEL_NAME

# ── Map array task index → (SFT checkpoint, GRPO data selection) ──────────────
SFT_SELS=("diverse_5pct" "random_5pct" "diverse_20pct" "random_20pct")
GRPO_SELS=("variance_5pct" "random_5pct" "variance_20pct" "random_20pct")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SFT_IDX=$((IDX / 4))
GRPO_IDX=$((IDX % 4))
SFT_SEL=${SFT_SELS[$SFT_IDX]}
GRPO_SEL=${GRPO_SELS[$GRPO_IDX]}

echo "Array task $IDX: SFT=$SFT_SEL  GRPO=$GRPO_SEL  MODEL=$MODEL_NAME"

mkdir -p $LOGS_DIR $CKPT_DIR/grpo

module load apptainer/1.4.2

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

EXTRA_ARGS=""
[ -n "$DRY_RUN" ] && EXTRA_ARGS="--dry-run"

cd $REPO_DIR
singularity exec --nv \
    --overlay $OVERLAY:ro \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    $SIF \
    python3 scripts/04_train_grpo.py \
        --seed $SEED \
        --data-dir $REPO_DIR/data \
        --sft-checkpoints-dir $CKPT_DIR/sft \
        --grpo-checkpoints-dir $CKPT_DIR/grpo \
        --logs-dir $LOGS_DIR \
        --sft-selections "$SFT_SEL" \
        --grpo-selections "$GRPO_SEL" \
        $EXTRA_ARGS
