#!/usr/bin/env python3
"""
Phase 4 — Launch 16 GRPO training runs via verl's main_ppo with GRPO estimator.

For each of the 4 SFT checkpoints × 4 GRPO selections = 16 total runs.
Each run generates a per-run config and invokes verl as a subprocess.

Flags:
    --dry-run        caps each run to 20 training steps for pipeline smoke-testing.
    --sft-selections / --grpo-selections  subset of runs to execute.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeding import seed_everything

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
    p = argparse.ArgumentParser(description="Launch verl GRPO training runs.")
    p.add_argument("--config", default=str(ROOT / "configs" / "base_grpo.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--sft-checkpoints-dir", default="/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints/sft")
    p.add_argument("--grpo-checkpoints-dir", default="/orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints/grpo")
    p.add_argument("--logs-dir", default=str(ROOT / "logs"))
    p.add_argument("--sft-selections", nargs="+", default=SFT_SELECTIONS)
    p.add_argument("--grpo-selections", nargs="+", default=GRPO_SELECTIONS)
    p.add_argument("--dry-run", action="store_true",
                   help="Cap each run to 20 steps for smoke testing.")
    return p.parse_args()


def load_base_config(config_path: Path) -> dict[str, str]:
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

    if any("." in k for k in raw):
        return {k: str(v) for k, v in raw.items()}
    return flatten(raw)


def build_verl_cmd(
    base_overrides: dict[str, str],
    run_overrides: dict[str, str],
    custom_reward_fn: str,
) -> list[str]:
    all_overrides = {**base_overrides, **run_overrides}
    kv_args = [f"{k}={v}" for k, v in all_overrides.items()]
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"reward_model.reward_manager={custom_reward_fn}",
    ] + kv_args
    return cmd


def stream_subprocess(cmd: list[str], log_path: Path) -> int:
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
    _setup_file_logger(Path(args.logs_dir), "04_train_grpo")

    data_dir = Path(args.data_dir)
    sft_ckpt_base = Path(args.sft_checkpoints_dir)
    grpo_ckpt_base = Path(args.grpo_checkpoints_dir)
    logs_dir = Path(args.logs_dir)

    base_config_path = Path(args.config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    base_overrides = load_base_config(base_config_path)
    reward_fn = "src.reward.gsm8k_reward.compute_score"

    failed = []
    total = len(args.sft_selections) * len(args.grpo_selections)
    count = 0

    for sft_sel in args.sft_selections:
        sft_ckpt = sft_ckpt_base / sft_sel
        if not sft_ckpt.exists():
            logger.warning("SFT checkpoint not found: %s — skipping.", sft_ckpt)
            continue

        for grpo_sel in args.grpo_selections:
            count += 1
            run_name = f"grpo_{sft_sel}_{grpo_sel}"
            train_parquet = data_dir / "grpo_train" / sft_sel / f"{grpo_sel}.parquet"

            if not train_parquet.exists():
                logger.warning(
                    "Missing GRPO train parquet: %s — skipping %s.", train_parquet, run_name
                )
                continue

            run_overrides = {
                "actor_rollout_ref.model.path": str(sft_ckpt),
                "data.train_files": str(train_parquet),
                "trainer.experiment_name": run_name,
                "trainer.default_local_dir": str(grpo_ckpt_base / sft_sel / grpo_sel),
            }
            if args.dry_run:
                run_overrides["trainer.total_training_steps"] = "20"

            cmd = build_verl_cmd(base_overrides, run_overrides, reward_fn)
            log_path = logs_dir / f"{run_name}.log"

            if args.dry_run:
                logger.info("[dry-run %d/%d] Would run: %s", count, total, " ".join(cmd))
                continue

            logger.info("=== Starting GRPO run %d/%d: %s ===", count, total, run_name)
            rc = stream_subprocess(cmd, log_path)
            if rc != 0:
                logger.error("GRPO run %s exited with code %d", run_name, rc)
                failed.append(run_name)
            else:
                logger.info("GRPO run %s finished successfully.", run_name)

    if failed:
        logger.error("The following GRPO runs FAILED: %s", failed)
        sys.exit(1)

    logger.info("Phase 4 complete. %d GRPO runs executed.", count)


if __name__ == "__main__":
    main()
