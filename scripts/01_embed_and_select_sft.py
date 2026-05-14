#!/usr/bin/env python3
"""
Phase 1 — Embed GSM8K train pool, run PCA, and select SFT subsets.

Outputs:
    data/embeddings.npy
    results/plots/pca_variance.png
    data/sft_indices/{diverse,random}_{5,20}pct.json
    data/sft_train/{diverse,random}_{5,20}pct.parquet
    results/plots/sft_selection_pca.png
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeding import seed_everything
from src.data.embed import embed_questions, fit_pca
from src.data.select_sft import diverse_select, random_select, quantile_uniform_select
from src.utils.plots import save_pca_variance_plot, save_sft_selection_pca_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BUDGETS = {"5pct": 0.05, "20pct": 0.20}


def _setup_file_logger(log_dir: Path, tag: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{tag}_{ts}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed and select SFT subsets.")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--variance-threshold", type=float, default=0.95)
    p.add_argument("--max-pca-components", type=int, default=50)
    p.add_argument(
        "--diversity-method",
        choices=["fps", "quantile-uniform"],
        default="fps",
        help="fps = farthest-point sampling (default); quantile-uniform = bin-based uniform.",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    _setup_file_logger(ROOT / "logs", "01_embed_and_select_sft")

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    # ── Load train parquet ───────────────────────────────────────────────────
    train_parquet = data_dir / "gsm8k_train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"{train_parquet} not found — run 00_prepare_gsm8k.py first.")

    import pandas as pd
    import numpy as np

    logger.info("Loading %s…", train_parquet)
    df = pd.read_parquet(train_parquet)
    questions = [row["question"] for row in df["extra_info"]]
    n_total = len(questions)
    logger.info("Pool size: %d examples", n_total)

    # ── Embed ────────────────────────────────────────────────────────────────
    embed_cache = data_dir / "embeddings.npy"
    embeddings = embed_questions(
        questions,
        model_name=args.embed_model,
        cache_path=embed_cache,
        force=args.force,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca_cache = data_dir / "pca_reduced.npy"
    pca_obj_cache = data_dir / "pca_model.pkl"

    if pca_cache.exists() and pca_obj_cache.exists() and not args.force:
        import pickle
        logger.info("Loading cached PCA from %s", pca_cache)
        reduced = np.load(pca_cache)
        with open(pca_obj_cache, "rb") as f:
            pca = pickle.load(f)
    else:
        import pickle
        pca, reduced = fit_pca(
            embeddings,
            variance_threshold=args.variance_threshold,
            max_components=args.max_pca_components,
        )
        np.save(pca_cache, reduced)
        with open(pca_obj_cache, "wb") as f:
            pickle.dump(pca, f)
        logger.info("Saved PCA reduced embeddings to %s", pca_cache)

    logger.info("Reduced shape: %s  (kept %d components)", reduced.shape, reduced.shape[1])

    # ── PCA variance plot ────────────────────────────────────────────────────
    pca_var_plot = results_dir / "plots" / "pca_variance.png"
    save_pca_variance_plot(pca.explained_variance_ratio_, pca_var_plot, args.variance_threshold)
    logger.info("Saved PCA variance plot: %s", pca_var_plot)

    # ── Selection ────────────────────────────────────────────────────────────
    indices_dir = data_dir / "sft_indices"
    sft_train_dir = data_dir / "sft_train"
    indices_dir.mkdir(parents=True, exist_ok=True)
    sft_train_dir.mkdir(parents=True, exist_ok=True)

    all_selections: dict[str, np.ndarray] = {}

    for budget_name, frac in BUDGETS.items():
        budget = max(1, int(round(n_total * frac)))
        logger.info("Budget %s: %d examples (%.1f%% of %d)", budget_name, budget, frac * 100, n_total)

        # Diverse selection.
        diverse_idx_path = indices_dir / f"diverse_{budget_name}.json"
        if diverse_idx_path.exists() and not args.force:
            logger.info("Loading cached diverse indices: %s", diverse_idx_path)
            diverse_idxs = np.array(json.loads(diverse_idx_path.read_text()))
        else:
            logger.info("Running diverse selection (%s)…", args.diversity_method)
            if args.diversity_method == "fps":
                diverse_idxs = diverse_select(reduced, budget, seed=args.seed)
            else:
                diverse_idxs = quantile_uniform_select(reduced, budget, seed=args.seed)
            diverse_idx_path.write_text(json.dumps(diverse_idxs.tolist()))
            logger.info("Saved diverse indices: %s", diverse_idx_path)

        # Random selection.
        random_idx_path = indices_dir / f"random_{budget_name}.json"
        if random_idx_path.exists() and not args.force:
            logger.info("Loading cached random indices: %s", random_idx_path)
            random_idxs = np.array(json.loads(random_idx_path.read_text()))
        else:
            random_idxs = random_select(n_total, budget, seed=args.seed)
            random_idx_path.write_text(json.dumps(random_idxs.tolist()))
            logger.info("Saved random indices: %s", random_idx_path)

        all_selections[f"diverse_{budget_name}"] = diverse_idxs
        all_selections[f"random_{budget_name}"] = random_idxs

        # Emit filtered parquets.
        for sel_name, idxs in [
            (f"diverse_{budget_name}", diverse_idxs),
            (f"random_{budget_name}", random_idxs),
        ]:
            out_parquet = sft_train_dir / f"{sel_name}.parquet"
            if out_parquet.exists() and not args.force:
                logger.info("Parquet already exists: %s", out_parquet)
                continue
            subset = df.iloc[idxs].reset_index(drop=True)
            subset.to_parquet(out_parquet, index=False)
            logger.info("Wrote %s  (%d rows)", out_parquet, len(subset))

    # ── Diagnostic scatter ───────────────────────────────────────────────────
    scatter_plot = results_dir / "plots" / "sft_selection_pca.png"
    save_sft_selection_pca_plot(reduced, all_selections, scatter_plot)
    logger.info("Saved SFT selection scatter: %s", scatter_plot)

    logger.info(
        "Phase 1 complete. %d selections saved to %s and %s.",
        len(all_selections),
        indices_dir,
        sft_train_dir,
    )


if __name__ == "__main__":
    main()
