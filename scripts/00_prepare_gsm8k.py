#!/usr/bin/env python3
"""
Phase 0 — Download openai/gsm8k and convert to verl-compatible parquet.

Outputs:
    data/gsm8k_train.parquet  (7,473 rows)
    data/gsm8k_test.parquet   (1,319 rows)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeding import seed_everything
from src.data.prepare_gsm8k import build_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _setup_file_logger(log_dir: Path, tag: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{tag}_{ts}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare GSM8K parquet files for verl.")
    p.add_argument("--config", default=None, help="Unused placeholder for uniform CLI interface.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"), help="Output directory for parquet files.")
    p.add_argument("--force", action="store_true", help="Recompute even if output files exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    _setup_file_logger(ROOT / "logs", "00_prepare_gsm8k")

    data_dir = Path(args.data_dir)
    train_out = data_dir / "gsm8k_train.parquet"
    test_out = data_dir / "gsm8k_test.parquet"

    if train_out.exists() and test_out.exists() and not args.force:
        logger.info("Parquet files already exist; use --force to recompute.")
        logger.info("  train: %s", train_out)
        logger.info("  test:  %s", test_out)
        return

    import pandas as pd
    from datasets import load_dataset

    logger.info("Loading openai/gsm8k (main)…")
    ds = load_dataset("openai/gsm8k", "main")

    logger.info("Building train rows (%d examples)…", len(ds["train"]))
    train_rows = build_rows(ds["train"], split_offset=0)
    train_df = pd.DataFrame(train_rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_out, index=False)
    logger.info("Wrote %s  (%d rows)", train_out, len(train_df))

    logger.info("Building test rows (%d examples)…", len(ds["test"]))
    test_rows = build_rows(ds["test"], split_offset=0)
    test_df = pd.DataFrame(test_rows)
    test_df.to_parquet(test_out, index=False)
    logger.info("Wrote %s  (%d rows)", test_out, len(test_df))

    logger.info(
        "Phase 0 complete. Train=%d rows, Test=%d rows.",
        len(train_df),
        len(test_df),
    )


if __name__ == "__main__":
    main()
