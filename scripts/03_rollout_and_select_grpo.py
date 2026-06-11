#!/usr/bin/env python3
"""
Phase 3 — Score train-pool candidates with rollouts and select GRPO subsets.

For each of the four SFT checkpoints:
  1. Generate 5 rollouts per candidate using vLLM.
  2. Record per-example: rewards list, mean, std, pass@1, pass@5.
  3. Cache to data/rollouts/{sft_run_name}.jsonl.
  4. Select top-k% by reward std (variance selection) and matching random baselines.
  5. Emit filtered parquets to data/grpo_train/{sft_run_name}/{selection_name}.parquet.
  6. Save reward scatter plot to results/plots/grpo_reward_scatter_{sft_run_name}.png.

Outputs:
    data/rollouts/{sft_run_name}.jsonl
    data/grpo_train/{sft_run_name}/{variance,random}_{10,20}pct.parquet
    results/plots/grpo_reward_scatter_{sft_run_name}.png
    results/grpo_selection_stats.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeding import seed_everything
from src.utils.logging import setup_file_logger
from src.data.select_grpo import variance_select, random_select
from src.utils.plots import save_grpo_reward_scatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SFT_SELECTIONS = ["diverse_10pct", "random_10pct", "diverse_20pct", "random_20pct"]
GRPO_BUDGETS = {"10pct": 0.10, "20pct": 0.20}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout scoring and GRPO subset selection.")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints" / "sft"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--logs-dir", default=str(ROOT / "logs"))
    p.add_argument("--sft-selections", nargs="+", default=SFT_SELECTIONS)
    p.add_argument("--candidate-cap", type=int, default=2000,
                   help="Max candidates to score per SFT ckpt (0 = use full train set).")
    p.add_argument("--n-rollouts", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_rollout_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    logger.info("Loaded %d cached rollout results from %s", len(results), path)
    return results


def save_rollout_cache(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info("Cached %d rollout results to %s", len(results), path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    setup_file_logger(Path(args.logs_dir), "03_rollout_and_select_grpo")

    import numpy as np
    import pandas as pd

    data_dir = Path(args.data_dir)
    ckpt_base = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)

    train_parquet = data_dir / "gsm8k_train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"{train_parquet} not found — run 00_prepare_gsm8k.py first.")

    train_df = pd.read_parquet(train_parquet)
    n_total = len(train_df)

    # Cap candidate pool if requested.
    cap = args.candidate_cap if args.candidate_cap > 0 else n_total
    cap = min(cap, n_total)
    rng = np.random.default_rng(args.seed)
    candidate_idxs = rng.choice(n_total, size=cap, replace=False)
    candidate_df = train_df.iloc[candidate_idxs].reset_index(drop=True)
    logger.info("Candidate pool: %d examples", len(candidate_df))

    prompts = candidate_df["prompt"].tolist()
    ground_truths = [row["ground_truth"] for row in candidate_df["reward_model"]]

    all_stats: dict[str, dict] = {}

    for sft_sel in args.sft_selections:
        # Prefer the best checkpoint saved by Phase 2; fall back to the raw dir.
        best_ckpt = ckpt_base / sft_sel / "best"
        raw_ckpt = ckpt_base / sft_sel
        if best_ckpt.exists():
            ckpt_path = best_ckpt
            logger.info("Using best checkpoint for %s: %s", sft_sel, ckpt_path)
        elif raw_ckpt.exists():
            ckpt_path = raw_ckpt
            logger.warning(
                "No 'best' checkpoint found for %s; falling back to %s. "
                "Run Phase 2 with the updated script to enable best-model selection.",
                sft_sel, ckpt_path,
            )
        else:
            logger.warning("Checkpoint not found for %s: %s — skipping.", sft_sel, raw_ckpt)
            continue

        rollout_cache = data_dir / "rollouts" / f"{sft_sel}.jsonl"

        if rollout_cache.exists() and not args.force:
            results = load_rollout_cache(rollout_cache)
        else:
            logger.info("Scoring pool with SFT checkpoint: %s", ckpt_path)
            from src.rollout.score_pool import score_pool
            results = score_pool(
                model_path=str(ckpt_path),
                prompts=prompts,
                ground_truths=ground_truths,
                n_rollouts=args.n_rollouts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=args.seed,
            )
            save_rollout_cache(results, rollout_cache)

        std_rewards = np.array([r["std_reward"] for r in results])
        mean_rewards = np.array([r["mean_reward"] for r in results])

        grpo_train_base = data_dir / "grpo_train" / sft_sel
        grpo_train_base.mkdir(parents=True, exist_ok=True)

        sft_stats: dict[str, dict] = {}
        selection_masks: dict[str, np.ndarray] = {}

        for budget_name, frac in GRPO_BUDGETS.items():
            budget = max(1, int(round(len(results) * frac)))
            logger.info(
                "[%s] GRPO budget %s: %d examples", sft_sel, budget_name, budget
            )

            # Variance selection.
            var_idx_path = data_dir / "rollouts" / f"{sft_sel}_variance_{budget_name}.json"
            if var_idx_path.exists() and not args.force:
                var_idxs = np.array(json.loads(var_idx_path.read_text()))
            else:
                var_idxs = variance_select(std_rewards, mean_rewards, budget, seed=args.seed)
                var_idx_path.write_text(json.dumps(var_idxs.tolist()))

            # Random selection.
            rand_idx_path = data_dir / "rollouts" / f"{sft_sel}_random_{budget_name}.json"
            if rand_idx_path.exists() and not args.force:
                rand_idxs = np.array(json.loads(rand_idx_path.read_text()))
            else:
                rand_idxs = random_select(len(results), budget, seed=args.seed)
                rand_idx_path.write_text(json.dumps(rand_idxs.tolist()))

            for sel_name, idxs in [
                (f"variance_{budget_name}", var_idxs),
                (f"random_{budget_name}", rand_idxs),
            ]:
                out_parquet = grpo_train_base / f"{sel_name}.parquet"
                if not out_parquet.exists() or args.force:
                    subset = candidate_df.iloc[idxs].reset_index(drop=True)
                    subset.to_parquet(out_parquet, index=False)
                    logger.info("Wrote %s  (%d rows)", out_parquet, len(subset))

                mask = np.zeros(len(results), dtype=bool)
                mask[idxs] = True
                selection_masks[sel_name] = mask

            sft_stats[f"variance_{budget_name}"] = {
                "mean_std": float(std_rewards[var_idxs].mean()),
                "mean_mean": float(mean_rewards[var_idxs].mean()),
            }
            sft_stats[f"random_{budget_name}"] = {
                "mean_std": float(std_rewards[rand_idxs].mean()),
                "mean_mean": float(mean_rewards[rand_idxs].mean()),
            }

        all_stats[sft_sel] = sft_stats

        # ── Reward scatter plot ───────────────────────────────────────────────
        scatter_path = results_dir / "plots" / f"grpo_reward_scatter_{sft_sel}.png"
        save_grpo_reward_scatter(
            mean_rewards,
            std_rewards,
            selection_masks,
            scatter_path,
            title=f"Reward distribution — SFT: {sft_sel}",
        )
        logger.info("Saved reward scatter: %s", scatter_path)

    # ── Save aggregated stats ─────────────────────────────────────────────────
    stats_path = results_dir / "grpo_selection_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(all_stats, indent=2))
    logger.info("Saved GRPO selection stats: %s", stats_path)

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
