# Command cheat sheet

All commands run from the repo root on the login node. Every artifact is
namespaced by `<MODEL_NAME>/seed<SEED>`, so different (model, seed) runs never
interfere.

## One command per full run (recommended)

Submits phases 0–4 as a SLURM dependency chain — each phase starts
automatically when the previous one succeeds:

```bash
make slurm-all MODEL=Qwen/Qwen3-1.7B SEED=42
make slurm-all MODEL=Qwen/Qwen3-1.7B SEED=43
make slurm-all MODEL=Qwen/Qwen3-1.7B SEED=44

make slurm-all MODEL=Qwen/Qwen3-4B SEED=42
make slurm-all MODEL=Qwen/Qwen3-4B SEED=43
make slurm-all MODEL=Qwen/Qwen3-4B SEED=44
```

All six chains can be submitted at the same time; SLURM queues them
independently. After a chain finishes, generate plots + CSVs (CPU, seconds):

```bash
make plot MODEL=Qwen/Qwen3-1.7B SEED=42   # …and so on per run
```

## Phase-by-phase (manual control)

Shown for Qwen3-1.7B seed 42 — change `MODEL=` / `SEED=` for other runs.
Each phase requires the previous one to have completed.

```bash
make slurm-prepare MODEL=Qwen/Qwen3-1.7B SEED=42   # Phase 0: GSM8K parquets (CPU, ~5 min)
make slurm-select  MODEL=Qwen/Qwen3-1.7B SEED=42   # Phase 1: SFT data selection (1 GPU, ~20 min)
make slurm-sft     MODEL=Qwen/Qwen3-1.7B SEED=42   # Phase 2: 4 SFT runs (array, 1 GPU each, ~1-2 h)
make slurm-rollout MODEL=Qwen/Qwen3-1.7B SEED=42   # Phase 3: rollout scoring (1 GPU, ~40 min)
make slurm-grpo    MODEL=Qwen/Qwen3-1.7B SEED=42   # Phase 4: 16 GRPO runs (array %2, ~2-4 h each)
make plot          MODEL=Qwen/Qwen3-1.7B SEED=42   # Phase 5: curves + CSVs (login node, seconds)
```

## Monitoring

```bash
squeue --me                      # job states; dependency-held jobs show (Dependency)
tail -f logs/Qwen3-1.7B/seed42/gsm8k_sft-<jobid>_0.out
```

If a phase fails, its dependents stay pinned in the queue with reason
`DependencyNeverSatisfied` — cancel them with `scancel <jobid>`, fix, and
resubmit from the failed phase onward (each phase caches its outputs, so
nothing completed is redone).

## Smoke test before a long run

```bash
DRY_RUN=1 MODEL=Qwen/Qwen3-1.7B MODEL_NAME=Qwen3-1.7B SEED=42 \
  sbatch scripts/slurm/submit_grpo.sh    # 20-step GRPO smoke test
```
