#!/bin/bash
#SBATCH -J gsm8k_embed
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -p mit_normal_gpu
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%j.out

# Phase 1 — embed + PCA + SFT subset selection.
# sentence-transformers benefits from GPU but runs on CPU too.
# Data selection is model-agnostic; results go to the model-specific results dir.
#
# Environment variables (set by Makefile or manually):
#   MODEL       full HuggingFace model ID  (default: Qwen/Qwen2.5-0.5B-Instruct)
#   MODEL_NAME  short name used in paths   (default: basename of MODEL)
#   SEED        random seed                (default: 42)

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img

MODEL=${MODEL:-"Qwen/Qwen2.5-0.5B-Instruct"}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
SEED=${SEED:-42}

LOGS_DIR=$REPO_DIR/logs/$MODEL_NAME
RESULTS=$REPO_DIR/results/$MODEL_NAME

mkdir -p $LOGS_DIR $RESULTS

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
        --results-dir $RESULTS
