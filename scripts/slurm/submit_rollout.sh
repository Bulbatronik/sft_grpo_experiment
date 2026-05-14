#!/bin/bash
#SBATCH -J gsm8k_rollout
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -p mit_normal_gpu
#SBATCH --output=%x-%j.out

# Phase 3 — rollout scoring + GRPO subset selection.
# Scores each SFT checkpoint against a candidate pool using vLLM.

SEED=${SEED:-42}
CANDIDATE_CAP=${CANDIDATE_CAP:-2000}
CKPT_DIR=/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints

module load apptainer

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

cd $HOME/gsm8k_selection_experiment
singularity exec --nv -B /orcd $HOME/verl.sif \
    python3 scripts/03_rollout_and_select_grpo.py \
        --seed $SEED \
        --candidate-cap $CANDIDATE_CAP \
        --checkpoints-dir $CKPT_DIR/sft
