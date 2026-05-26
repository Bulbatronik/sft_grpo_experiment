#!/bin/bash
#SBATCH -J gsm8k_embed
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -p mit_normal_gpu
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%j.out

# Phase 1 — embed + PCA + SFT subset selection
# sentence-transformers benefits from GPU but runs on CPU too.

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img
SEED=${SEED:-42}

mkdir -p $REPO_DIR/logs

module load apptainer/1.4.2

cd $REPO_DIR
singularity exec --nv \
    --overlay $OVERLAY \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    $SIF \
    python3 scripts/01_embed_and_select_sft.py \
        --seed $SEED \
        --data-dir $REPO_DIR/data \
        --results-dir $REPO_DIR/results
