#!/bin/bash
#SBATCH -J gsm8k_embed
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -p mit_normal_gpu
#SBATCH --output=%x-%j.out

# Phase 1 — embed + PCA + SFT subset selection
# sentence-transformers benefits from GPU but runs on CPU too.
module load apptainer

cd $HOME/gsm8k_selection_experiment
singularity exec --nv -B /orcd $HOME/verl.sif \
    python3 scripts/01_embed_and_select_sft.py --seed 42
