#!/bin/bash
#SBATCH -J gsm8k_prepare
#SBATCH -t 0:30:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -p mit_normal
#SBATCH --output=%x-%j.out

# Phase 0 — prepare GSM8K parquets (CPU only, no GPU needed)
eval "$(conda shell.bash hook)"
conda activate my_env

cd $HOME/gsm8k_selection_experiment
python scripts/00_prepare_gsm8k.py --seed 42
