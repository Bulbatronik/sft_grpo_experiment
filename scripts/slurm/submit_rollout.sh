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
CANDIDATE_CAP=${CANDIDATE_CAP:-0}   # 0 = score the full train pool

CKPT_DIR=$REPO_DIR/checkpoints/$MODEL_NAME
LOGS_DIR=$REPO_DIR/logs/$MODEL_NAME
RESULTS=$REPO_DIR/results/$MODEL_NAME

mkdir -p $LOGS_DIR

module load apptainer/1.4.2

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc
export VLLM_WORKER_MULTIPROC_METHOD=spawn

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
        --results-dir $RESULTS \
        --logs-dir $LOGS_DIR
