#!/bin/bash
#SBATCH -J gsm8k_eval
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -p mit_normal_gpu
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%j.out

# Phase 5 — evaluate all 21 models on the GSM8K test set.

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
CKPT_DIR=$REPO_DIR/checkpoints
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img
SEED=${SEED:-42}

mkdir -p $REPO_DIR/logs $REPO_DIR/results/eval

module load apptainer/1.4.2

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

cd $REPO_DIR
singularity exec --nv \
    --overlay $OVERLAY \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    $SIF \
    python3 scripts/05_evaluate.py \
        --seed $SEED \
        --data-dir $REPO_DIR/data \
        --sft-checkpoints-dir $CKPT_DIR/sft \
        --grpo-checkpoints-dir $CKPT_DIR/grpo \
        --results-dir $REPO_DIR/results \
        --logs-dir $REPO_DIR/logs
