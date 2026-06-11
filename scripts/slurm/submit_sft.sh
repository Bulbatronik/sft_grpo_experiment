#!/bin/bash
# ── SLURM resource request ────────────────────────────────────────────────────
#SBATCH -J gsm8k_sft
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH -p mit_preemptable
#SBATCH --array=0-3
#   ^─ 4 independent tasks, one per SFT data selection, all running in parallel.
#      Each task gets its own GPU and log file.
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%A_%a.out

# Phase 2 — SFT training (4 selections as a SLURM array).
#
# SLURM_ARRAY_TASK_ID maps to one SFT data selection:
#   0 → diverse_10pct
#   1 → random_10pct
#   2 → diverse_20pct
#   3 → random_20pct
#
# Environment variables (set by Makefile or manually):
#   MODEL       full HuggingFace model ID  (default: Qwen/Qwen3-1.7B)
#   MODEL_NAME  short name used in paths   (default: basename of MODEL)
#   SFT_CONFIG  path to SFT config YAML    (default: configs/sft_<MODEL_NAME>.yaml)
#   SEED        random seed                (default: 42)

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
SIF=/home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY=/home/usemil/orcd/scratch/apptainer/verl_overlay.img

MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
SEED=${SEED:-42}

RUN_DIR=$MODEL_NAME/seed$SEED
DATA_DIR=$REPO_DIR/data/$RUN_DIR
CKPT_DIR=$REPO_DIR/checkpoints/$RUN_DIR
LOGS_DIR=$REPO_DIR/logs/$RUN_DIR
RESULTS=$REPO_DIR/results/$RUN_DIR

# Resolve config: env var → model-specific config → base fallback
SFT_CONFIG=${SFT_CONFIG:-"$REPO_DIR/configs/sft_${MODEL_NAME}.yaml"}
[ -f "$SFT_CONFIG" ] || SFT_CONFIG="$REPO_DIR/configs/base_sft.yaml"

# ── Map array task index → SFT data selection ─────────────────────────────────
SFT_SELS=("diverse_10pct" "random_10pct" "diverse_20pct" "random_20pct")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SFT_SEL=${SFT_SELS[$IDX]}

echo "Array task $IDX: SFT=$SFT_SEL  MODEL=$MODEL_NAME  CONFIG=$(basename $SFT_CONFIG)"

mkdir -p $LOGS_DIR $CKPT_DIR/sft

module load apptainer/1.4.2

export CC=/usr/bin/gcc
export TRITON_CC=/usr/bin/gcc

cd $REPO_DIR
singularity exec --nv \
    --overlay $OVERLAY \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    $SIF \
    python3 scripts/02_train_sft.py \
        --config $SFT_CONFIG \
        model.name=$MODEL \
        --seed $SEED \
        --data-dir $DATA_DIR \
        --checkpoints-dir $CKPT_DIR/sft \
        --logs-dir $LOGS_DIR \
        --results-dir $RESULTS \
        --selections "$SFT_SEL"
