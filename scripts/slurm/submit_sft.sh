#!/bin/bash
#SBATCH -J gsm8k_sft
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH -p mit_preemptable # mit_normal_gpu
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%j.out

# Phase 2 — SFT training (4 runs, sequential).
# Usage: sbatch submit_sft.sh [selection_name] [dotlist_overrides...]
#   selection_name  : one of diverse_5pct, random_5pct, diverse_20pct, random_20pct
#                     (omit to run all four sequentially)
#   dotlist_overrides: OmegaConf-style overrides forwarded to the config, e.g.:
#                     sbatch submit_sft.sh "" training.lr=1e-4 training.max_steps=30

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
CKPT_DIR=$REPO_DIR/checkpoints
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img
SELECTION=${1:-""}
SEED=${SEED:-42}
# All remaining positional args are forwarded as dotlist overrides.
shift 1 2>/dev/null; OVERRIDES="$@"

mkdir -p $REPO_DIR/logs $CKPT_DIR/sft

module load apptainer/1.4.2

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

SELECTION_ARG=""
[ -n "$SELECTION" ] && SELECTION_ARG="--selections $SELECTION"

cd $REPO_DIR
singularity exec --nv \
    --overlay $OVERLAY \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    $SIF \
    python3 scripts/02_train_sft.py \
        --config $REPO_DIR/configs/base_sft.yaml \
        --seed $SEED \
        --data-dir $REPO_DIR/data \
        --checkpoints-dir $CKPT_DIR/sft \
        --logs-dir $REPO_DIR/logs \
        --results-dir $REPO_DIR/results \
        $SELECTION_ARG \
        $OVERRIDES
