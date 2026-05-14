#!/usr/bin/env python3
"""
Phase 5 — Evaluate all 21 models on the GSM8K test set.

Models evaluated:
  1. Base model (Qwen/Qwen2.5-0.5B-Instruct)
  4. SFT models (one per SFT selection)
  16. GRPO models (4 SFT × 4 GRPO selections)

Outputs per model: results/eval/{model_id}.json
Summary table:    results/summary.csv, results/summary.md
Bar chart:        results/plots/final_accuracy.png
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeding import seed_everything
from src.eval.test_eval import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SFT_SELECTIONS = ["diverse_5pct", "random_5pct", "diverse_20pct", "random_20pct"]
GRPO_SELECTIONS = ["variance_5pct", "random_5pct", "variance_20pct", "random_20pct"]


def _setup_file_logger(log_dir: Path, tag: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{tag}_{ts}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all models on GSM8K test set.")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--sft-checkpoints-dir", default="/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints/sft")
    p.add_argument("--grpo-checkpoints-dir", default="/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints/grpo")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--logs-dir", default=str(ROOT / "logs"))
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model IDs to evaluate (default: all).",
    )
    return p.parse_args()


def run_eval(
    model_id: str,
    model_path: str,
    test_parquet: Path,
    out_path: Path,
    gpu_util: float,
    seed: int,
    force: bool,
) -> dict | None:
    if out_path.exists() and not force:
        logger.info("Cached eval result found: %s", out_path)
        return json.loads(out_path.read_text())

    if not Path(model_path).exists() and "/" not in model_path:
        logger.warning("Model path not found: %s — skipping.", model_path)
        return None

    logger.info("Evaluating: %s  (%s)", model_id, model_path)
    try:
        result = evaluate_model(
            model_path=model_path,
            test_parquet=test_parquet,
            gpu_memory_utilization=gpu_util,
            seed=seed,
        )
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", model_id, e)
        return None

    result["model_id"] = model_id
    result["model_path"] = model_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved eval result: %s  (greedy acc=%.4f)", out_path, result["greedy_accuracy"])
    return result


def render_summary_md(rows: list[dict], out_path: Path) -> None:
    sft_keys = ["random_5pct", "diverse_5pct", "random_20pct", "diverse_20pct"]
    grpo_keys = ["random_5pct", "variance_5pct", "random_20pct", "variance_20pct"]

    lines = [
        "# GSM8K Selection Experiment — Results",
        "",
        "## 4×4 Accuracy Table (greedy decoding)",
        "",
        "| SFT \\ GRPO | random 5% | variance 5% | random 20% | variance 20% |",
        "|---|---|---|---|---|",
    ]

    grpo_acc: dict[tuple, float] = {}
    for row in rows:
        if row.get("sft_sel") and row.get("grpo_sel"):
            grpo_acc[(row["sft_sel"], row["grpo_sel"])] = row["greedy_accuracy"]

    for sk in sft_keys:
        cells = [f"**SFT: {sk}**"]
        for gk in grpo_keys:
            acc = grpo_acc.get((sk, gk))
            cells.append(f"{acc:.4f}" if acc is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Reference lines", ""]
    for row in rows:
        if row.get("type") in ("base", "sft_only"):
            lines.append(f"- **{row['model_id']}**: {row['greedy_accuracy']:.4f}")

    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote summary markdown: %s", out_path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    _setup_file_logger(Path(args.logs_dir), "05_evaluate")

    data_dir = Path(args.data_dir)
    sft_ckpt_base = Path(args.sft_checkpoints_dir)
    grpo_ckpt_base = Path(args.grpo_checkpoints_dir)
    results_dir = Path(args.results_dir)
    eval_dir = results_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_parquet = data_dir / "gsm8k_test.parquet"
    if not test_parquet.exists():
        raise FileNotFoundError(f"{test_parquet} not found — run 00_prepare_gsm8k.py first.")

    # Build the full model list.
    model_list: list[dict] = []

    model_list.append(
        {"model_id": "base", "model_path": args.base_model, "type": "base",
         "sft_sel": None, "grpo_sel": None}
    )

    for sft_sel in SFT_SELECTIONS:
        ckpt = sft_ckpt_base / sft_sel
        model_list.append(
            {"model_id": f"sft_{sft_sel}", "model_path": str(ckpt), "type": "sft_only",
             "sft_sel": sft_sel, "grpo_sel": None}
        )

    for sft_sel in SFT_SELECTIONS:
        for grpo_sel in GRPO_SELECTIONS:
            ckpt = grpo_ckpt_base / sft_sel / grpo_sel
            model_list.append(
                {
                    "model_id": f"grpo_{sft_sel}_{grpo_sel}",
                    "model_path": str(ckpt),
                    "type": "grpo",
                    "sft_sel": sft_sel,
                    "grpo_sel": grpo_sel,
                }
            )

    if args.models:
        model_list = [m for m in model_list if m["model_id"] in args.models]

    all_results = []
    for entry in model_list:
        out_path = eval_dir / f"{entry['model_id']}.json"
        result = run_eval(
            model_id=entry["model_id"],
            model_path=entry["model_path"],
            test_parquet=test_parquet,
            out_path=out_path,
            gpu_util=args.gpu_memory_utilization,
            seed=args.seed,
            force=args.force,
        )
        if result is not None:
            row = {**entry, **{k: v for k, v in result.items() if k != "per_example"}}
            all_results.append(row)

    if not all_results:
        logger.warning("No evaluation results collected.")
        return

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = results_dir / "summary.csv"
    fieldnames = ["model_id", "type", "sft_sel", "grpo_sel", "greedy_accuracy", "pass_at_1_sampled"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Saved summary CSV: %s", csv_path)

    # ── Save summary markdown ─────────────────────────────────────────────────
    md_path = results_dir / "summary.md"
    render_summary_md(all_results, md_path)

    # ── Final accuracy bar chart ──────────────────────────────────────────────
    base_acc = next(
        (r["greedy_accuracy"] for r in all_results if r["type"] == "base"), 0.0
    )
    sft_only_accs = {
        r["sft_sel"]: r["greedy_accuracy"]
        for r in all_results
        if r["type"] == "sft_only" and r["sft_sel"]
    }
    grpo_results = {
        (r["sft_sel"], r["grpo_sel"]): r["greedy_accuracy"]
        for r in all_results
        if r["type"] == "grpo"
    }

    if grpo_results:
        from src.utils.plots import save_final_accuracy_bar
        bar_path = results_dir / "plots" / "final_accuracy.png"
        save_final_accuracy_bar(grpo_results, base_acc, sft_only_accs, bar_path)
        logger.info("Saved final accuracy bar chart: %s", bar_path)

    logger.info("Phase 5 complete. Results in %s", results_dir)


if __name__ == "__main__":
    main()
