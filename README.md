# GSM8K Selection Experiment

Investigating two data-selection hypotheses for math reasoning fine-tuning on
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) using
[verl](https://github.com/volcengine/verl) (v0.3+):

1. **SFT diversity hypothesis** — Embedding-diverse training examples (broad
   coverage of the question space) outperform random selection for SFT.
2. **GRPO variance hypothesis** — Examples with high reward variance across
   rollouts provide more learning signal than random selection for GRPO.

All 16 combinations of (SFT selection × GRPO selection) are compared at 5% and
20% data budgets against random baselines, plus a base model and SFT-only
reference lines.

---

## Hypotheses

| # | Stage | Selection | Hypothesis |
|---|-------|-----------|------------|
| 1 | SFT   | Diverse (FPS in embedding space) | Covers more of the question distribution; better generalisation |
| 2 | SFT   | Random | Baseline |
| 3 | GRPO  | High reward variance (≈ 50% pass rate) | Maximum advantage signal; more effective policy gradient |
| 4 | GRPO  | Random | Baseline |

---

## Models

| Role | Identifier |
|------|-----------|
| Base / SFT / GRPO | `Qwen/Qwen2.5-0.5B-Instruct` |
| Embedding (diversity) | `sentence-transformers/all-MiniLM-L6-v2` |

---

## Install

See [INSTALLATION.md](INSTALLATION.md) for the full setup guide (Singularity container, conda env, overlay).

Verified stack: `verl 0.8.0.dev` · `vllm 0.11.0` · `torch 2.8.0+cu128`

---

## Running the pipeline

Jobs are submitted via SLURM from the login node. Each phase depends on the previous one completing successfully.

```bash
cd /home/usemil/orcd/scratch/sft_grpo_experiment

# Phase 0 — data prep (CPU, ~5 min)
make slurm-prepare

# Phase 1 — embed + PCA + SFT selection (~20 min, 1 GPU)
make slurm-embed

# Phase 2 — 4 SFT runs (~2-4 h each, 1 GPU)
make slurm-sft

# Phase 3 — rollout scoring + GRPO selection (~1-2 h, 1 GPU)
make slurm-rollout

# Phase 4 — 16 GRPO runs (SLURM array, 4 concurrent, ~1-2 h each)
make slurm-grpo

# Phase 5 — evaluate all 21 models (~1-2 h, 1 GPU)
make slurm-eval
```

Monitor jobs with `squeue --me`. Logs go to `logs/`.

All intermediate outputs are cached; re-running a phase is safe. Use `--force` to recompute.

### Smoke-testing GRPO

```bash
# Run 20 steps only to check configs before committing hours of compute
DRY_RUN=1 sbatch scripts/slurm/submit_grpo.sh
```

---

## Expected runtime (single A100 80 GB)

| Phase | Approximate time |
|-------|-----------------|
| 0 – Data preparation | < 5 min |
| 1 – Embedding + PCA + selection | 10–20 min |
| 2 – SFT × 4 runs | 2–4 h each (3 epochs, ~7k examples max) |
| 3 – Rollout scoring (2,000 candidates × 4 ckpts × 5 rollouts) | 1–2 h |
| 4 – GRPO × 16 runs | 1–2 h each (1 epoch) |
| 5 – Evaluation (21 models × 1,319 test examples) | 1–2 h total |

Total end-to-end: **~3–5 days** of sequential GPU time. Runs can be parallelised
across multiple GPUs by setting `NPROC`.

---

## Outputs

```
results/
├── plots/
│   ├── pca_variance.png          # Explained variance curve
│   ├── sft_selection_pca.png     # PC1×PC2 scatter of all selections
│   ├── sft_losses.png            # Training loss across 4 SFT runs
│   ├── grpo_reward_scatter_*.png # mean vs std reward scatter per SFT ckpt
│   └── final_accuracy.png        # Grouped bar chart (main result)
├── eval/
│   └── {model_id}.json           # Per-model accuracy + per-example correctness
├── grpo_selection_stats.json     # Reward distribution stats for GRPO selections
├── summary.csv                   # Machine-readable 4×4 table
└── summary.md                    # Human-readable results table
```

### Reading `results/summary.md`

The 4×4 table shows **greedy test accuracy** for each (SFT strategy, GRPO
strategy) combination. Higher is better. Reference lines at the bottom show
the base model and SFT-only accuracies.

A cell that is **notably higher than its row's random baseline** supports the
GRPO variance hypothesis for that SFT initialisation; a column that is higher
than the random column supports the SFT diversity hypothesis for that GRPO
regime.

---

## Directory layout

```
sft_grpo_experiment/
├── configs/
│   ├── base_sft.yaml      shared SFT hyperparameters
│   ├── base_grpo.yaml     shared GRPO hyperparameters
│   └── runs/              per-run generated overrides
├── data/
│   ├── gsm8k_{train,test}.parquet
│   ├── embeddings.npy
│   ├── pca_{reduced.npy,model.pkl}
│   ├── sft_indices/       selected index JSON files
│   ├── sft_train/         filtered SFT parquets
│   ├── rollouts/          JSONL rollout caches + index files
│   └── grpo_train/        filtered GRPO parquets
├── src/
│   ├── data/              prepare_gsm8k, embed, select_sft, select_grpo
│   ├── reward/            gsm8k_reward (shared between rollout + verl)
│   ├── rollout/           score_pool (vLLM inference)
│   ├── eval/              test_eval
│   └── utils/             seeding, plots
├── scripts/               00–05 pipeline scripts
├── checkpoints/           SFT and GRPO model weights
├── logs/                  per-run log files
└── results/               plots, eval JSONs, summary table
```
