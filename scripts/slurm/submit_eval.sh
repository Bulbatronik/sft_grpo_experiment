#!/bin/bash
#SBATCH -J gsm8k_eval
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -p mit_normal_gpu
#SBATCH --output=%x-%j.out

# Phase 5 — evaluate all 21 models on the GSM8K test set.

SEED=${SEED:-42}
CKPT_DIR=/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints

module load apptainer

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

cd $HOME/gsm8k_selection_experiment
singularity exec --nv -B /orcd $HOME/verl.sif \
    python3 scripts/05_evaluate.py \
        --seed $SEED \
        --sft-checkpoints-dir $CKPT_DIR/sft \
        --grpo-checkpoints-dir $CKPT_DIR/grpo
