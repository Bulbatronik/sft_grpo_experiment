#!/bin/bash
#SBATCH -J gsm8k_grpo
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -p mit_normal_gpu
#SBATCH --array=0-15%4
#SBATCH --output=%x-%A_%a.out

# Phase 4 — GRPO training (16 runs via SLURM array, up to 4 concurrent).
# Array index 0-15 maps to all 16 (sft_sel, grpo_sel) combinations.

SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-""}

SFT_SELS=("diverse_5pct" "random_5pct" "diverse_20pct" "random_20pct")
GRPO_SELS=("variance_5pct" "random_5pct" "variance_20pct" "random_20pct")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SFT_IDX=$((IDX / 4))
GRPO_IDX=$((IDX % 4))
SFT_SEL=${SFT_SELS[$SFT_IDX]}
GRPO_SEL=${GRPO_SELS[$GRPO_IDX]}

CKPT_DIR=/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints

echo "Array task $IDX: SFT=$SFT_SEL  GRPO=$GRPO_SEL"

module load apptainer

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

cd $HOME/gsm8k_selection_experiment

EXTRA_ARGS=""
if [ -n "$DRY_RUN" ]; then
    EXTRA_ARGS="--dry-run"
fi

singularity exec --nv -B /orcd $HOME/verl.sif \
    python3 scripts/04_train_grpo.py \
        --seed $SEED \
        --sft-checkpoints-dir $CKPT_DIR/sft \
        --grpo-checkpoints-dir $CKPT_DIR/grpo \
        --sft-selections "$SFT_SEL" \
        --grpo-selections "$GRPO_SEL" \
        $EXTRA_ARGS
