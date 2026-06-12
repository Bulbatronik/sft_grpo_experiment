#!/bin/bash
#SBATCH -J gsm8k_select
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -p mit_normal_gpu
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%j.out

# Phase 1 — SFT subset selection (pluggable strategies).
# The diverse strategy embeds with sentence-transformers (GPU helps, CPU works);
# complexity strategies (ops/sentences) need no GPU at all.
#
# Environment variables (set by Makefile or manually):
#   MODEL       full HuggingFace model ID  (default: Qwen/Qwen3-1.7B)
#   MODEL_NAME  short name used in paths   (default: basename of MODEL)
#   SEED        random seed                (default: 42)
#   STRATEGIES  selection strategies       (default: "diverse random")
# All paths are namespaced by <MODEL_NAME>/seed<SEED>.

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img

MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
SEED=${SEED:-42}
STRATEGIES=${STRATEGIES:-"diverse random"}

RUN_DIR=$MODEL_NAME/seed$SEED
DATA_DIR=$REPO_DIR/data/$RUN_DIR
LOGS_DIR=$REPO_DIR/logs/$RUN_DIR
RESULTS=$REPO_DIR/results/$RUN_DIR

mkdir -p $LOGS_DIR $RESULTS

module load apptainer/1.4.2

cd $REPO_DIR
singularity exec --nv \
    --overlay $OVERLAY:ro \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    $SIF \
    python3 scripts/01_select_sft.py \
        --seed $SEED \
        --data-dir $DATA_DIR \
        --results-dir $RESULTS \
        --strategies $STRATEGIES \
        --ifd-model $MODEL
