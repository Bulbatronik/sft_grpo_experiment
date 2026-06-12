#!/usr/bin/env python3
"""
Phase 1 — Select SFT training subsets from the GSM8K pool.

Selection strategies are pluggable (see src/data/select_sft.py):
    diverse    — embed questions (MiniLM) → PCA (variance threshold) →
                 farthest-point sampling. Embeddings/PCA are computed only
                 when this strategy is requested.
    random     — uniform random baseline.
    ops        — most complex first, by number of <<...>> intermediate
                 calculations in the ground-truth solution.
    sentences  — most complex first, by number of reasoning sentences.
    ifd_hard / ifd_easy / ifd_mid
               — Instruction-Following Difficulty: perplexity of the solution
                 given the question vs alone, scored with --ifd-model (the
                 model you will fine-tune). hard = highest IFD, easy = lowest,
                 mid = band around the median. Scores cached per model.

Run with --strategies to choose which selections to produce, e.g.:
    python3 scripts/01_select_sft.py --strategies diverse random ifd_hard

Outputs (per strategy s and budget b):
    <data-dir>/sft_indices/<s>_<b>.json
    <data-dir>/sft_train/<s>_<b>.parquet
    <results-dir>/plots/pca_variance.png        (when embeddings are computed)
    <results-dir>/plots/sft_selection_pca.png   (one scatter per selection)
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
from src.data.embed import embed_questions, fit_pca
from src.data.select_sft import STRATEGIES, select
from src.utils.plots import save_pca_variance_plot, save_sft_selection_pca_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BUDGETS = {"10pct": 0.10, "20pct": 0.20}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select SFT training subsets.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument(
        "--strategies", nargs="+", default=["diverse", "random"],
        choices=sorted(STRATEGIES),
        help="Selection strategies to run (default: diverse random).",
    )
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--variance-threshold", type=float, default=0.95,
                   help="PCA keeps the fewest components reaching this "
                        "cumulative explained variance (no component cap).")
    p.add_argument("--ifd-model", default="Qwen/Qwen3-1.7B",
                   help="Model used to score IFD (should be the model you "
                        "will fine-tune). Only loaded for ifd_* strategies.")
    p.add_argument("--ifd-batch-size", type=int, default=8)
    p.add_argument("--force", action="store_true",
                   help="Recompute embeddings/PCA/selections even if cached.")
    return p.parse_args()


def _compute_reduced(args, questions, data_dir, results_dir):
    """Embed → PCA, with file caching. Only called when a strategy needs it."""
    import numpy as np
    import pickle

    embeddings = embed_questions(
        questions,
        model_name=args.embed_model,
        cache_path=data_dir / "embeddings.npy",
        force=args.force,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)

    pca_cache = data_dir / "pca_reduced.npy"
    pca_obj_cache = data_dir / "pca_model.pkl"

    if pca_cache.exists() and pca_obj_cache.exists() and not args.force:
        logger.info("Loading cached PCA from %s", pca_cache)
        reduced = np.load(pca_cache)
        with open(pca_obj_cache, "rb") as f:
            pca = pickle.load(f)
    else:
        pca, reduced = fit_pca(embeddings, variance_threshold=args.variance_threshold)
        np.save(pca_cache, reduced)
        with open(pca_obj_cache, "wb") as f:
            pickle.dump(pca, f)
        logger.info("Saved PCA reduced embeddings to %s", pca_cache)

    logger.info("Reduced shape: %s  (kept %d components)", reduced.shape, reduced.shape[1])

    plot_path = results_dir / "plots" / "pca_variance.png"
    save_pca_variance_plot(pca.explained_variance_ratio_, plot_path, args.variance_threshold)
    logger.info("Saved PCA variance plot: %s", plot_path)

    return reduced


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    setup_file_logger(ROOT / "logs", "01_select_sft")

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

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
    logger.info("Strategies: %s", args.strategies)

    # Gather only the inputs the requested strategies actually need.
    needs = set().union(*(STRATEGIES[s]["needs"] for s in args.strategies))
    reduced = _compute_reduced(args, questions, data_dir, results_dir) if "reduced" in needs else None
    solutions = (
        [row["answer"] for row in df["extra_info"]]
        if needs & {"solutions", "ifd_scores"} else None
    )

    ifd_scores = None
    if "ifd_scores" in needs:
        from src.data.ifd import compute_ifd_scores
        from src.data.prepare_gsm8k import _SYSTEM_PROMPT
        ifd_model_short = args.ifd_model.split("/")[-1]
        ifd_scores = compute_ifd_scores(
            questions, solutions,
            model_name=args.ifd_model,
            system_prompt=_SYSTEM_PROMPT,
            cache_path=data_dir / f"ifd_scores_{ifd_model_short}.npy",
            force=args.force,
            batch_size=args.ifd_batch_size,
        )

    indices_dir = data_dir / "sft_indices"
    sft_train_dir = data_dir / "sft_train"
    indices_dir.mkdir(parents=True, exist_ok=True)
    sft_train_dir.mkdir(parents=True, exist_ok=True)

    all_selections: dict[str, np.ndarray] = {}

    for budget_name, frac in BUDGETS.items():
        budget = max(1, int(round(n_total * frac)))
        logger.info("Budget %s: %d examples (%.1f%% of %d)", budget_name, budget, frac * 100, n_total)

        for strategy in args.strategies:
            sel_name = f"{strategy}_{budget_name}"
            idx_path = indices_dir / f"{sel_name}.json"

            if idx_path.exists() and not args.force:
                logger.info("Loading cached indices: %s", idx_path)
                idxs = np.array(json.loads(idx_path.read_text()))
            else:
                logger.info("Running %s selection…", strategy)
                idxs = select(
                    strategy, n_total, budget, seed=args.seed,
                    reduced=reduced, solutions=solutions, ifd_scores=ifd_scores,
                )
                idx_path.write_text(json.dumps(idxs.tolist()))
                logger.info("Saved indices: %s", idx_path)

            all_selections[sel_name] = idxs

            out_parquet = sft_train_dir / f"{sel_name}.parquet"
            if out_parquet.exists() and not args.force:
                logger.info("Parquet already exists: %s", out_parquet)
                continue
            subset = df.iloc[idxs].reset_index(drop=True)
            subset.to_parquet(out_parquet, index=False)
            logger.info("Wrote %s  (%d rows)", out_parquet, len(subset))

    # Diagnostic PCA scatters (only meaningful when embeddings exist).
    if reduced is not None:
        plots_dir = results_dir / "plots"
        save_sft_selection_pca_plot(reduced, all_selections, plots_dir)
        logger.info("Saved SFT selection scatters to %s", plots_dir)

    logger.info(
        "Phase 1 complete. %d selections saved to %s and %s.",
        len(all_selections), indices_dir, sft_train_dir,
    )


if __name__ == "__main__":
    main()
