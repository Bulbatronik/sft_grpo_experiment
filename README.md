# GSM8K Selection Experiment

Investigating two data-selection hypotheses for math reasoning fine-tuning on
[GSM8K](https://huggingface.co/datasets/openai/gsm8k):

1. **SFT diversity hypothesis** — Embedding-diverse training examples (broad
   coverage of the question space) outperform random selection for SFT.
2. **GRPO variance hypothesis** — Examples with high reward variance across
   rollouts provide more learning signal than random selection for GRPO.

All 16 combinations of (SFT selection × GRPO selection) are compared at **10%
and 20% data budgets** against random baselines. The pipeline produces a
branching-tree accuracy plot: SFT training curves that branch into 4 GRPO
curves at the best SFT checkpoint.

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
| Primary | `Qwen/Qwen3-1.7B` (default) |
| Larger  | `Qwen/Qwen3-4B` |
| Embedding (diversity) | `sentence-transformers/all-MiniLM-L6-v2` |

Both SFT (TRL) and GRPO (verl) use **LoRA rank 32** to keep checkpoint sizes
small and make the pipeline feasible on a single GPU. Switching models only
requires passing `MODEL=<hf_id>`.

**Every artifact — data selections, checkpoints, logs, results — is namespaced
by `<MODEL_NAME>/seed<SEED>`** (default seed 42). Data selections are
seed-dependent (random subsets differ per seed), so the data directory is part
of the namespace too: re-running the pipeline with `SEED=43` produces a fully
independent replicate without touching the seed-42 artifacts. This is what
enables mean±std reporting across seeds for the paper.

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
make slurm-embed MODEL=Qwen/Qwen3-1.7B

# Phase 2 — 4 SFT runs as a SLURM array (~1–2 h each, 1 GPU per task)
make slurm-sft MODEL=Qwen/Qwen3-1.7B

# Phase 3 — rollout scoring + GRPO selection (~40 min, 1 GPU)
make slurm-rollout MODEL=Qwen/Qwen3-1.7B

# Phase 4 — 16 GRPO runs as a SLURM array (2 concurrent, ~2–4 h each)
make slurm-grpo MODEL=Qwen/Qwen3-1.7B

# Phase 5 — plot accuracy training curves (CPU, seconds)
make plot MODEL=Qwen/Qwen3-1.7B
```

To run with the 4B model, change `MODEL=Qwen/Qwen3-4B` — the pipeline picks
up `configs/sft_Qwen3-4B.yaml` and `configs/grpo_Qwen3-4B.yaml` automatically.

To run a replicate with another seed, add `SEED=43` to every command
(including `slurm-prepare`, since the data selections are seed-dependent).

Monitor jobs with `squeue --me`. Logs go to `logs/<MODEL_NAME>/seed<SEED>/`.

All intermediate outputs are cached; re-running a phase is safe.

### Smoke-testing GRPO

```bash
# Run 20 steps only to verify configs before committing hours of compute
make grpo-dry MODEL=Qwen/Qwen3-1.7B
```

---

## Expected runtime (1 GPU, mit_preemptable / mit_normal_gpu)

| Phase | Approximate time |
|-------|-----------------|
| 0 – Data preparation | < 5 min |
| 1 – Embedding + PCA + selection | 10–20 min |
| 2 – SFT × 4 runs | 1–2 h per run (200 steps max, early stopping on val accuracy) |
| 3 – Rollout scoring | ~40 min total |
| 4 – GRPO × 16 runs | 2–4 h each (300 steps, val accuracy logged every 20 steps) |
| 5 – Plot | < 1 min |

Total end-to-end: ~2–4 days of sequential GPU time. Phases 2 and 4 run as
SLURM arrays, so wall-clock time is reduced to the longest single task.

---

## Outputs

```
results/<MODEL_NAME>/seed<SEED>/
├── sft_metrics_<sel>.csv             # per-step train loss, grad norm, lr, eval loss/acc
│                                     # (written by Phase 2; one file per array task)
├── sft_eval_metrics.csv              # eval accuracy per SFT step, parsed from logs (Phase 5)
├── grpo_metrics.csv                  # per-step reward, pg/kl loss, entropy, grad norm,
│                                     # response length, val accuracy (Phase 5)
└── plots/
    ├── pca_variance.png              # Explained variance curve
    ├── sft_selection_pca.png         # PC1×PC2 scatter of all selections
    ├── sft_losses_<sel>.png          # Train + val CE-loss curves (per array task)
    ├── sft_accuracy_<sel>.png        # Val accuracy curve (per array task)
    ├── grpo_reward_scatter_*.png     # mean vs std reward scatter per SFT ckpt
    ├── curves_<sft_sel>.png          # SFT→GRPO accuracy curve per SFT selection
    └── curves_all.png                # All 4 SFT + 16 GRPO branches combined
```

The CSVs let you track overfitting and training dynamics without re-parsing
logs: `sft_metrics_*.csv` has the dense per-step loss/grad-norm trace from the
Trainer itself, while `grpo_metrics.csv` is parsed from verl console logs
(reward mean, policy-gradient loss, KL loss, entropy, response length per step,
plus test accuracy every `test_freq` steps).

The `curves_*.png` plots are the primary result. The x-axis shows real
training steps: SFT eval points on the left (every 10 steps, up to early
stopping), GRPO eval points to the right (every 20 steps, offset by the best
SFT step for each selection). The combined plot uses a fixed x-axis range
(0–500: max 200 SFT + max 300 GRPO) so runs with different early-stopping
steps remain visually comparable.

Color = SFT data selection; line style = GRPO data selection; ★ = best SFT
checkpoint (GRPO branch point).

Validation accuracy during GRPO is logged directly by verl against the GSM8K
test set (`data/gsm8k_test.parquet`) every 20 steps — no separate evaluation
pass is needed.

---

## Configs

```
configs/
├── base_sft.yaml           fallback SFT config (Qwen2.5-0.5B, LoRA rank 64)
├── base_grpo.yaml          fallback GRPO config (Qwen2.5-0.5B, LoRA rank 64)
├── sft_Qwen3-1.7B.yaml     SFT for Qwen3-1.7B  (rank 32, lr 1e-4, 200 steps)
├── sft_Qwen3-4B.yaml       SFT for Qwen3-4B    (rank 32, lr 5e-5, 200 steps)
├── grpo_Qwen3-1.7B.yaml    GRPO for Qwen3-1.7B (rank 32, n=8, 300 steps)
└── grpo_Qwen3-4B.yaml      GRPO for Qwen3-4B   (rank 32, n=8, 300 steps, smaller batches)
```

Config files are auto-selected from `MODEL_NAME` (e.g. `MODEL=Qwen/Qwen3-1.7B`
→ `configs/sft_Qwen3-1.7B.yaml`). Override with `SFT_CONFIG=` or
`GRPO_CONFIG=` if needed.

---

## Checkpoint layout

```
checkpoints/<MODEL_NAME>/seed<SEED>/
├── sft/
│   └── <sft_sel>/
│       ├── best/          ← LoRA adapter (PEFT format, saved by trainer.save_model())
│       ├── best_merged/   ← merged full model (loaded by Phase 4 as GRPO start)
│       └── best_step.txt
└── grpo/
    └── <sft_sel>/<grpo_sel>/
        └── global_step_<N>/   ← verl checkpoint (last step only, max_actor_ckpt_to_keep=1)
```

`best_merged/` is created by `merge_and_unload()` at the end of Phase 2. Phase
4 prefers `best_merged/` and falls back to `best/` (full fine-tune case where
`lora.rank=0`).

---

## Directory layout

```
sft_grpo_experiment/
├── configs/               per-model SFT + GRPO hyperparameter files
├── data/
│   └── <MODEL_NAME>/seed<SEED>/   per-run data dir (selections are seed-dependent)
│       ├── gsm8k_{train,test}.parquet
│       ├── embeddings.npy
│       ├── pca_{reduced.npy,model.pkl}
│       ├── sft_indices/       selected index JSON files
│       ├── sft_train/         filtered SFT parquets
│       ├── rollouts/          JSONL rollout caches + index files
│       └── grpo_train/        filtered GRPO parquets
├── src/
│   ├── data/              prepare_gsm8k, embed, select_sft, select_grpo
│   ├── reward/            gsm8k_reward (shared between rollout + verl)
│   ├── rollout/           score_pool (vLLM inference)
│   └── utils/             seeding, plots
├── scripts/               00–04, 07 pipeline scripts + slurm/ submission scripts
├── checkpoints/           SFT and GRPO model weights (per MODEL_NAME/seedSEED)
├── logs/                  per-run SLURM log files (per MODEL_NAME/seedSEED)
└── results/               plots + metric CSVs (per MODEL_NAME/seedSEED)
```

---

## Makefile variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-1.7B` | Full HuggingFace model ID |
| `MODEL_NAME` | `$(notdir $(MODEL))` | Short name used in paths |
| `SFT_CONFIG` | `configs/sft_$(MODEL_NAME).yaml` | SFT hyperparameter file |
| `GRPO_CONFIG` | `configs/grpo_$(MODEL_NAME).yaml` | GRPO hyperparameter file |
| `SEED` | `42` | Global random seed; part of every artifact path |
| `RUN_DIR` | `$(MODEL_NAME)/seed$(SEED)` | Namespace for data/checkpoints/logs/results |
| `SIF` | `/home/usemil/orcd/scratch/apptainer/verl.sif` | Singularity image |
| `OVERLAY` | `/home/usemil/orcd/scratch/apptainer/verl_overlay.img` | Overlay image |
