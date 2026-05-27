#!/usr/bin/env python3
"""
Phase 2 — Launch four SFT training runs via verl's fsdp_sft_trainer.

For each of the four selections (diverse/random × 5%/20%), generates a per-run
config and invokes verl as a subprocess, streaming output to both the console
and a per-run log file.

After all runs complete, parses verl's console logs to produce a training-loss
comparison plot at results/plots/sft_losses.png.
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeding import seed_everything
from src.utils.logging import setup_file_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SELECTIONS = ["diverse_5pct", "random_5pct", "diverse_20pct", "random_20pct"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch verl SFT training runs.")
    p.add_argument("--config", default=str(ROOT / "configs" / "base_sft.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--checkpoints-dir", default="/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints/sft")
    p.add_argument("--logs-dir", default=str(ROOT / "logs"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--nproc", type=int, default=1, help="GPUs per node.")
    p.add_argument(
        "--selections",
        nargs="+",
        default=SELECTIONS,
        help="Which selections to train (default: all four).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return p.parse_args()


def load_base_config(config_path: Path) -> dict[str, str]:
    """Parse YAML-ish key: value file into flat dict of CLI overrides."""
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    def flatten(d: dict, prefix: str = "") -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, key + "."))
            else:
                out[key] = str(v)
        return out

    # The base_sft.yaml uses dotted keys directly (not nested YAML), so handle both.
    if any("." in k for k in raw):
        return {k: str(v) for k, v in raw.items()}
    return flatten(raw)


def build_verl_cmd(
    base_overrides: dict[str, str],
    run_overrides: dict[str, str],
    nproc: int,
) -> list[str]:
    all_overrides = {**base_overrides, **run_overrides}
    kv_args = [f"{k}={v}" for k, v in all_overrides.items()]
    # verl 0.8.x: entry point is verl.trainer.sft_trainer (renamed from fsdp_sft_trainer)
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        "-m", "verl.trainer.sft_trainer",
    ] + kv_args
    return cmd


def parse_loss_from_log(log_path: Path) -> list[float]:
    """Extract training loss values from a verl SFT log file."""
    losses = []
    # verl logs lines like: train/loss: 1.2345  or  'loss': 1.2345
    pattern = re.compile(r"(?:train/loss|'loss')\s*[=:]\s*([0-9]+\.[0-9]+)")
    if not log_path.exists():
        return losses
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            losses.append(float(m.group(1)))
    return losses


def stream_subprocess(cmd: list[str], log_path: Path) -> int:
    """Run cmd, stream stdout+stderr to console and log_path. Return exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("CMD: %s", " ".join(cmd))
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()
    return proc.returncode


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    setup_file_logger(Path(args.logs_dir), "02_train_sft")

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    logs_dir = Path(args.logs_dir)
    results_dir = Path(args.results_dir)

    base_config_path = Path(args.config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    base_overrides = load_base_config(base_config_path)

    failed = []
    for sel in args.selections:
        train_parquet = data_dir / "sft_train" / f"{sel}.parquet"
        if not train_parquet.exists():
            logger.warning("Missing parquet for %s: %s — skipping.", sel, train_parquet)
            continue

        run_overrides = {
            "data.train_files": str(train_parquet),
            "trainer.experiment_name": f"sft_{sel}",
            "trainer.default_local_dir": str(ckpt_dir / sel),
        }

        cmd = build_verl_cmd(base_overrides, run_overrides, args.nproc)
        log_path = logs_dir / f"sft_{sel}.log"

        if args.dry_run:
            logger.info("[dry-run] Would run: %s", " ".join(cmd))
            continue

        logger.info("=== Starting SFT run: %s ===", sel)
        rc = stream_subprocess(cmd, log_path)
        if rc != 0:
            logger.error("SFT run %s exited with code %d", sel, rc)
            failed.append(sel)
        else:
            logger.info("SFT run %s finished successfully.", sel)

    if args.dry_run:
        return

    # ── Plot training losses ──────────────────────────────────────────────────
    loss_curves: dict[str, list[float]] = {}
    for sel in args.selections:
        log_path = logs_dir / f"sft_{sel}.log"
        losses = parse_loss_from_log(log_path)
        if losses:
            loss_curves[sel] = losses
        else:
            logger.warning("No loss values found in %s", log_path)

    if loss_curves:
        from src.utils.plots import save_sft_loss_plot
        out_plot = results_dir / "plots" / "sft_losses.png"
        save_sft_loss_plot(loss_curves, out_plot)
        logger.info("Saved loss comparison plot: %s", out_plot)

    if failed:
        logger.error("The following runs FAILED: %s", failed)
        sys.exit(1)

    logger.info("Phase 2 complete. All SFT runs finished.")


if __name__ == "__main__":
    main()
