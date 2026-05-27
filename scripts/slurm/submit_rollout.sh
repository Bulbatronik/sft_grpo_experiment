#!/bin/bash
#SBATCH -J gsm8k_rollout
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -p mit_normal_gpu
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%j.out

# Phase 3 — rollout scoring + GRPO subset selection.
# Scores each SFT checkpoint against a candidate pool using vLLM.

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
CKPT_DIR=$REPO_DIR/checkpoints
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img
SEED=${SEED:-42}
CANDIDATE_CAP=${CANDIDATE_CAP:-0}   # 0 = score the full train pool (7,473 examples)

mkdir -p $REPO_DIR/logs

module load apptainer/1.4.2

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

cd $REPO_DIR
singularity exec --nv \
    --overlay $OVERLAY \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    $SIF \
    python3 scripts/03_rollout_and_select_grpo.py \
        --seed $SEED \
        --candidate-cap $CANDIDATE_CAP \
        --data-dir $REPO_DIR/data \
        --checkpoints-dir $CKPT_DIR/sft \
        --results-dir $REPO_DIR/results \
        --logs-dir $REPO_DIR/logs
