# GSM8K Selection Experiment

Investigating two data-selection hypotheses for math reasoning fine-tuning on
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) using
[verl](https://github.com/volcengine/verl) (v0.3+):

1. **SFT diversity hypothesis** — Embedding-diverse training examples (broad
   coverage of the question space) outperform random selection for SFT.
2. **GRPO variance hypothesis** — Examples with high reward variance across
   rollouts provide more learning signal than random selection for GRPO.

All 16 combinations of (SFT selection × GRPO selection) are compared at 5% and
20% data budgets against random baselines.

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

SFT uses LoRA (rank 64) via TRL; GRPO uses LoRA (rank 64) via verl FSDP.
This keeps checkpoint sizes small and makes the pipeline feasible for larger
models (e.g. 8B) without changing the code — only `model.name` and batch
sizes in the configs need updating.

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

# Phase 2 — 4 SFT runs (~40 min each, 1 GPU)
make slurm-sft

# Phase 3 — rollout scoring + GRPO selection (~40 min, 1 GPU)
make slurm-rollout

# Phase 4 — 16 GRPO runs (SLURM array, 4 concurrent, ~1-2 h each)
make slurm-grpo

# Phase 5 — plot accuracy training curves (CPU, seconds)
make plot
```

Monitor jobs with `squeue --me`. Logs go to `logs/`.

All intermediate outputs are cached; re-running a phase is safe.

### Smoke-testing GRPO

```bash
# Run 20 steps only to check configs before committing hours of compute
make grpo-dry
```

---

## Expected runtime (observed on mit_preemptable/mit_normal_gpu, 1 GPU)

| Phase | Approximate time |
|-------|-----------------|
| 0 – Data preparation | < 5 min |
| 1 – Embedding + PCA + selection | 10–20 min |
| 2 – SFT × 4 runs | ~40 min per run (~2h total, 50 steps, early stopping by val accuracy) |
| 3 – Rollout scoring (7,473 candidates × 4 ckpts × 5 rollouts) | ~40 min total |
| 4 – GRPO × 16 runs | ~1–2 h each (50 steps, val accuracy logged every 10 steps) |
| 5 – Plot | < 1 min |

Total end-to-end: **~1–2 days** of sequential GPU time.

---

## Outputs

```
results/
└── plots/
    ├── pca_variance.png              # Explained variance curve
    ├── sft_selection_pca.png         # PC1×PC2 scatter of all selections
    ├── sft_losses.png                # Training loss across 4 SFT runs
    ├── grpo_reward_scatter_*.png     # mean vs std reward scatter per SFT ckpt
    ├── curves_<sft_sel>.png          # SFT→GRPO accuracy curve per SFT selection
    └── curves_all.png                # All 4 SFT + 16 GRPO branches combined
```

The `curves_*.png` plots are the primary result. X-axis 0–100: SFT training
steps on the left, GRPO steps on the right (branching from the best SFT
checkpoint). Color = SFT data selection; line style = GRPO data selection.

Validation accuracy during GRPO is logged directly to `logs/` by verl against
the GSM8K test set (`data/gsm8k_test.parquet`) every 10 steps — no separate
evaluation pass is needed.

---

## Checkpoint layout

```
checkpoints/
├── sft/
│   └── <sft_sel>/
│       ├── best/          ← LoRA adapter (PEFT format)
│       ├── best_merged/   ← full merged model (used as GRPO starting point)
│       └── best_step.txt
└── grpo/
    └── <sft_sel>/<grpo_sel>/
        └── global_step_<N>/   ← verl checkpoint (base + LoRA, last step only)
```

---

## Directory layout

```
sft_grpo_experiment/
├── configs/
│   ├── base_sft.yaml      shared SFT hyperparameters (LoRA rank, lr, steps)
│   ├── base_grpo.yaml     shared GRPO hyperparameters (LoRA rank, batch, steps)
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
│   └── utils/             seeding, plots
├── scripts/               00–04, 07 pipeline scripts
├── checkpoints/           SFT and GRPO model weights
├── logs/                  per-run log files
└── results/               plots
```
