#!/bin/bash
# Submit the full pipeline (phases 0–4) as a SLURM dependency chain.
# Run from the login node — this script only calls sbatch, no compute happens
# here. Each phase starts automatically when the previous one finishes
# successfully (afterok), so no babysitter job is needed.
#
# Usage (or via `make slurm-all MODEL=... SEED=...`):
#   MODEL=Qwen/Qwen3-1.7B SEED=42 bash scripts/slurm/submit_all.sh
#
# Environment variables:
#   MODEL        full HuggingFace model ID  (default: Qwen/Qwen3-1.7B)
#   MODEL_NAME   short name used in paths   (default: basename of MODEL)
#   SEED         random seed                (default: 42)
#   SFT_CONFIG   SFT config YAML            (default: configs/sft_<MODEL_NAME>.yaml)
#   GRPO_CONFIG  GRPO config YAML           (default: configs/grpo_<MODEL_NAME>.yaml)

set -euo pipefail

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
SLURM_DIR=$REPO_DIR/scripts/slurm

MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
SEED=${SEED:-42}
SFT_CONFIG=${SFT_CONFIG:-$REPO_DIR/configs/sft_${MODEL_NAME}.yaml}
GRPO_CONFIG=${GRPO_CONFIG:-$REPO_DIR/configs/grpo_${MODEL_NAME}.yaml}
STRATEGIES=${STRATEGIES:-"diverse random"}

RUN_DIR=$MODEL_NAME/seed$SEED
DATA_DIR=$REPO_DIR/data/$RUN_DIR
LOGS_DIR=$REPO_DIR/logs/$RUN_DIR

mkdir -p "$LOGS_DIR"

# sbatch propagates the caller's environment to the jobs by default.
export MODEL MODEL_NAME SEED DATA_DIR SFT_CONFIG GRPO_CONFIG STRATEGIES

echo "Submitting pipeline for MODEL=$MODEL  SEED=$SEED"
echo "  logs → $LOGS_DIR"

j_prep=$(sbatch --parsable --output=$LOGS_DIR/%x-%j.out \
         $SLURM_DIR/submit_prepare.sh)
echo "  phase 0 prepare:  job $j_prep"

j_select=$(sbatch --parsable --dependency=afterok:$j_prep \
           --output=$LOGS_DIR/%x-%j.out $SLURM_DIR/submit_select.sh)
echo "  phase 1 select:   job $j_select  (after $j_prep)"

j_sft=$(sbatch --parsable --dependency=afterok:$j_select \
        --output=$LOGS_DIR/%x-%A_%a.out $SLURM_DIR/submit_sft.sh)
echo "  phase 2 sft:      job $j_sft  (array 0-3, after $j_select)"

j_roll=$(sbatch --parsable --dependency=afterok:$j_sft \
         --output=$LOGS_DIR/%x-%j.out $SLURM_DIR/submit_rollout.sh)
echo "  phase 3 rollout:  job $j_roll  (after all sft tasks)"

j_grpo=$(sbatch --parsable --dependency=afterok:$j_roll \
         --output=$LOGS_DIR/%x-%A_%a.out $SLURM_DIR/submit_grpo.sh)
echo "  phase 4 grpo:     job $j_grpo  (array 0-15%2, after $j_roll)"

echo ""
echo "Monitor with:  squeue --me"
echo "When phase 4 finishes, plot with:"
echo "  make plot MODEL=$MODEL SEED=$SEED"
