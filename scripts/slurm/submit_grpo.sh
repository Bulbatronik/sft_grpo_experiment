#!/bin/bash
# ── SLURM resource request ────────────────────────────────────────────────────
#SBATCH -J gsm8k_grpo                  # job name (shown in squeue)
#SBATCH -t 6:00:00                     # max wall time per array task
#SBATCH --gres=gpu:1                   # 1 GPU per task
#SBATCH --mem=64G                      # RAM per task
#SBATCH -c 16                          # CPU cores per task
#SBATCH -p mit_normal_gpu              # partition / queue
#SBATCH --array=0-15%4                 # 16 tasks (0–15), at most 4 running at once
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%A_%a.out
#                                        %x=job name  %A=array job ID  %a=task index

# Phase 4 — GRPO training (16 runs via SLURM array, up to 4 concurrent).
# Array index 0-15 maps to all 16 (sft_sel, grpo_sel) combinations.
# To smoke-test: DRY_RUN=1 sbatch submit_grpo.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
CKPT_DIR=$REPO_DIR/checkpoints
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif       # container image
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img  # writable env overlay

# ── Runtime options ───────────────────────────────────────────────────────────
SEED=${SEED:-42}       # override with: SEED=1 sbatch submit_grpo.sh
DRY_RUN=${DRY_RUN:-""} # override with: DRY_RUN=1 sbatch submit_grpo.sh

# ── Map array task index → (SFT checkpoint, GRPO data selection) ──────────────
# Layout: rows = SFT selections (index / 4), cols = GRPO selections (index % 4)
# e.g. task 0 → (diverse_5pct, variance_5pct), task 15 → (random_20pct, random_20pct)
SFT_SELS=("diverse_5pct" "random_5pct" "diverse_20pct" "random_20pct")
GRPO_SELS=("variance_5pct" "random_5pct" "variance_20pct" "random_20pct")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SFT_IDX=$((IDX / 4))
GRPO_IDX=$((IDX % 4))
SFT_SEL=${SFT_SELS[$SFT_IDX]}
GRPO_SEL=${GRPO_SELS[$GRPO_IDX]}

echo "Array task $IDX: SFT=$SFT_SEL  GRPO=$GRPO_SEL"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p $REPO_DIR/logs $CKPT_DIR/grpo

module load apptainer/1.4.2

# Tell Triton (used internally by vllm/torch) which C compiler to use.
export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

EXTRA_ARGS=""
if [ -n "$DRY_RUN" ]; then
    EXTRA_ARGS="--dry-run"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# --nv              pass GPU through to the container
# --overlay         mount the writable conda/pip environment
# -B /orcd,/home    bind-mount host directories so data and checkpoints are visible
# PYTHONNOUSERSITE  ignore ~/.local packages; use only what's in the overlay
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
        --logs-dir $REPO_DIR/logs \
        --sft-selections "$SFT_SEL" \
        --grpo-selections "$GRPO_SEL" \
        $EXTRA_ARGS
