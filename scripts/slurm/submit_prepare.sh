#!/bin/bash
#SBATCH -J gsm8k_prepare
#SBATCH -t 0:30:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -p mit_normal
#SBATCH --output=/home/usemil/orcd/scratch/sft_grpo_experiment/logs/%x-%j.out

# Phase 0 — prepare GSM8K parquets (CPU only, no GPU needed)

REPO_DIR=/home/usemil/orcd/scratch/sft_grpo_experiment
SEED=${SEED:-42}

mkdir -p $REPO_DIR/logs

source /etc/profile.d/modules.sh
module load miniforge
conda activate dataval_env

cd $REPO_DIR
python3 scripts/00_prepare_gsm8k.py --seed $SEED --data-dir $REPO_DIR/data
