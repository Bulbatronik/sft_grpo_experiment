#!/bin/bash
#SBATCH -J gsm8k_sft
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -p mit_normal_gpu
#SBATCH --output=%x-%j.out

# Phase 2 — SFT training (4 runs, sequential).
# Each run is passed as a positional argument; runs all four by default.
# Usage: sbatch submit_sft.sh [selection_name]

SELECTION=${1:-""}  # empty = run all four
SEED=${SEED:-42}
CKPT_DIR=/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints

module load apptainer

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

cd $HOME/gsm8k_selection_experiment

if [ -n "$SELECTION" ]; then
    singularity exec --nv -B /orcd $HOME/verl.sif \
        python3 scripts/02_train_sft.py --seed $SEED --nproc 1 \
            --checkpoints-dir $CKPT_DIR/sft \
            --selections "$SELECTION"
else
    singularity exec --nv -B /orcd $HOME/verl.sif \
        python3 scripts/02_train_sft.py --seed $SEED --nproc 1 \
            --checkpoints-dir $CKPT_DIR/sft
fi
