#!/usr/bin/env python3
"""
Phase 2 — SFT training via TRL's SFTTrainer.

Hyperparameters are read from configs/base_sft.yaml (via OmegaConf).
Individual values can be overridden from the command line using dotlist syntax:

    python3 scripts/02_train_sft.py training.lr=1e-4 eval.gen_batch_size=16

Infrastructure arguments (paths, seed, selections) are passed as regular flags.

New capabilities over the previous verl-based version:
  - eval_on_start: evaluates before any training step (step 0)
  - EarlyStoppingCallback: stops when eval loss stops improving
  - GSM8KAccuracyCallback: greedy-decodes the val set at every eval step
    and reports exact-match accuracy against the ground-truth #### answer

Output plots (results/plots/):
  sft_losses.png    — train + val CE-loss curves
  sft_accuracy.png  — val accuracy curve per selection

Checkpoint layout (compatible with Phase 4):
  checkpoints/sft/<sel>/best/   ← HuggingFace model saved here
  checkpoints/sft/<sel>/best_step.txt
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reward.gsm8k_reward import compute_score
from src.utils.logging import setup_file_logger
from src.utils.seeding import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SELECTIONS = ["diverse_5pct", "random_5pct", "diverse_20pct", "random_20pct"]
_DEFAULT_CONFIG = ROOT / "configs" / "base_sft.yaml"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    """
    Returns (args, cfg) where:
      args — infrastructure flags (paths, seed, selections, dry-run)
      cfg  — OmegaConf DictConfig loaded from YAML, merged with any dotlist
             overrides passed as extra positional arguments.
    """
    from omegaconf import OmegaConf

    p = argparse.ArgumentParser(
        description="SFT training via TRL SFTTrainer.",
        epilog="Any extra args are treated as OmegaConf dotlist overrides, "
               "e.g. training.lr=1e-4",
    )
    p.add_argument("--config", default=str(_DEFAULT_CONFIG),
                   help="Path to the YAML hyperparameter config.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints" / "sft"))
    p.add_argument("--logs-dir", default=str(ROOT / "logs"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--selections", nargs="+", default=SELECTIONS,
                   help="Which SFT subsets to train (default: all four).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config without training.")

    args, overrides = p.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return args, cfg


# ── Accuracy callback ─────────────────────────────────────────────────────────

from transformers import TrainerCallback  # noqa: E402 — needed for class definition


class GSM8KAccuracyCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that runs greedy generation on the full GSM8K
    validation set at every evaluation step (including step 0) and logs exact-
    match accuracy against the ground-truth #### answer.

    Results are accumulated in self.history as (step, accuracy) pairs and used
    after training to generate the accuracy plot.
    """

    def __init__(self, eval_dataset, tokenizer, max_new_tokens: int, batch_size: int):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.history: list[tuple[int, float]] = []

        # Pre-format prompt strings and collect ground truths at construction
        # time so on_evaluate only needs to run generation.
        self.prompts: list[str] = []
        self.ground_truths: list[str] = []
        for ex in eval_dataset:
            self.prompts.append(
                tokenizer.apply_chat_template(
                    ex["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            self.ground_truths.append(ex["reward_model"]["ground_truth"])

        logger.info("GSM8KAccuracyCallback ready: %d eval examples.", len(self.prompts))

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        import torch

        was_training = model.training
        model.eval()
        # Generation requires left-padding; restore original side afterwards.
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        device = next(model.parameters()).device
        correct = 0.0
        total = len(self.prompts)

        try:
            for start in tqdm(range(0, total, self.batch_size), desc="gsm8k_eval", leave=False):
                batch_prompts = self.prompts[start : start + self.batch_size]
                batch_gts = self.ground_truths[start : start + self.batch_size]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).to(device)

                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # With left-padding all sequences share the same prompt offset.
                prompt_len = inputs.input_ids.shape[1]
                for gen_ids, gt in zip(out_ids, batch_gts):
                    response = self.tokenizer.decode(
                        gen_ids[prompt_len:], skip_special_tokens=True
                    )
                    correct += compute_score(response, gt)
        finally:
            self.tokenizer.padding_side = orig_padding_side
            if was_training:
                model.train()

        accuracy = correct / total
        self.history.append((state.global_step, accuracy))
        logger.info(
            "step:%d - eval/gsm8k_accuracy:%.4f (%d/%d correct)",
            state.global_step, accuracy, int(correct), total,
        )


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_parquet_as_hf(path: Path):
    import pandas as pd
    from datasets import Dataset

    df = pd.read_parquet(path)
    return Dataset.from_pandas(df, preserve_index=False)


def _add_text_column(dataset, tokenizer):
    """Apply the tokenizer's chat template to the 'messages' column → 'text'."""
    def _fmt(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
    return dataset.map(_fmt, desc="chat template")


# ── Per-run training ──────────────────────────────────────────────────────────

def run_sft(sel: str, args, cfg, tokenizer) -> dict:
    """
    Train one SFT selection with TRL's SFTTrainer.

    Returns a dict with keys:
        'train'    : list[(step, loss)]
        'val'      : list[(step, loss)]
        'accuracy' : list[(step, accuracy)]
    """
    import torch
    from transformers import AutoModelForCausalLM, EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(args.data_dir)
    sel_ckpt_dir = Path(args.checkpoints_dir) / sel
    intermediate_dir = sel_ckpt_dir / "checkpoints"
    best_dir = sel_ckpt_dir / "best"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds_raw = _load_parquet_as_hf(data_dir / "sft_train" / f"{sel}.parquet")
    val_ds_raw = _load_parquet_as_hf(data_dir / "gsm8k_test.parquet")

    train_ds = _add_text_column(train_ds_raw, tokenizer)
    val_ds = _add_text_column(val_ds_raw, tokenizer)

    # Drop all columns except 'text': newer TRL checks for a 'prompt' column
    # before honouring dataset_text_field, hijacking the code path.
    train_ds = train_ds.select_columns(["text"])
    val_ds = val_ds.select_columns(["text"])

    logger.info("[%s] train=%d  val=%d", sel, len(train_ds), len(val_ds))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    accuracy_cb = GSM8KAccuracyCallback(
        eval_dataset=val_ds_raw,
        tokenizer=tokenizer,
        max_new_tokens=cfg.eval.max_new_tokens,
        batch_size=cfg.eval.gen_batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    tokenizer.model_max_length = cfg.model.max_seq_length
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = None
    if cfg.lora.rank > 0:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            task_type="CAUSAL_LM",
        )
        logger.info("[%s] LoRA enabled: rank=%d alpha=%d target=%s",
                    sel, cfg.lora.rank, cfg.lora.alpha, cfg.lora.target_modules)

    # ── Trainer ───────────────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=str(intermediate_dir),
        # Steps — effective batch = per_device × grad_accum × 1 GPU
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # Optimiser
        learning_rate=cfg.training.lr,
        lr_scheduler_type=cfg.training.lr_scheduler,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        # Precision and memory
        bf16=True,
        gradient_checkpointing=True,
        # Logging
        logging_steps=1,
        report_to="none",
        # Evaluation & checkpointing
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Data
        dataset_text_field="text",
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience
            ),
            accuracy_cb,
        ],
    )

    trainer.train()

    # ── Save best model ────────────────────────────────────────────────────────
    # load_best_model_at_end has already restored the best weights into the
    # trainer's model; save them in HF format for Phase 4 (GRPO).
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # When LoRA was used, trainer.save_model() saves only the adapter.
    # Also save a merged (adapter-free) full model so Phase 4 can start GRPO
    # from a clean base without needing to load + merge at runtime.
    if peft_config is not None:
        merged_dir = sel_ckpt_dir / "best_merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info("[%s] Merged LoRA checkpoint saved → %s", sel, merged_dir)

    best_ckpt_path = trainer.state.best_model_checkpoint
    if best_ckpt_path:
        best_step = int(Path(best_ckpt_path).name.split("-")[-1])
        (sel_ckpt_dir / "best_step.txt").write_text(str(best_step))
        logger.info("[%s] Best step: %d → %s", sel, best_step, best_dir)

    # Remove intermediate per-step checkpoints now that best/ is saved.
    if intermediate_dir.exists():
        shutil.rmtree(intermediate_dir)
        logger.info("[%s] Removed intermediate checkpoints.", sel)

    # ── Extract metrics ────────────────────────────────────────────────────────
    train_losses: list[tuple[int, float]] = []
    val_losses: list[tuple[int, float]] = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            train_losses.append((entry["step"], entry["loss"]))
        if "eval_loss" in entry:
            val_losses.append((entry["step"], entry["eval_loss"]))

    accuracy_history = list(accuracy_cb.history)

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {"train": train_losses, "val": val_losses, "accuracy": accuracy_history}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args, cfg = parse_args()
    seed_everything(args.seed)
    setup_file_logger(Path(args.logs_dir), "02_train_sft")

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    if args.dry_run:
        logger.info("selections=%s  seed=%d", args.selections, args.seed)
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: dict[str, dict] = {}
    failed: list[str] = []
    for sel in args.selections:
        train_parquet = Path(args.data_dir) / "sft_train" / f"{sel}.parquet"
        if not train_parquet.exists():
            logger.warning("Missing parquet for %s — skipping.", sel)
            continue
        logger.info("=== Starting SFT run: %s ===", sel)
        try:
            results[sel] = run_sft(sel, args, cfg, tokenizer)
            logger.info("=== Finished SFT run: %s ===", sel)
        except Exception:
            logger.exception("SFT run %s failed.", sel)
            failed.append(sel)

    # ── Plots ─────────────────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)

    train_curves = {s: d["train"] for s, d in results.items() if d.get("train")}
    val_curves   = {s: d["val"]   for s, d in results.items() if d.get("val")}
    acc_curves   = {s: d["accuracy"] for s, d in results.items() if d.get("accuracy")}

    from src.utils.plots import save_sft_accuracy_plot, save_sft_loss_plot

    if train_curves or val_curves:
        out = results_dir / "plots" / "sft_losses.png"
        save_sft_loss_plot(train_curves, val_curves, out)
        logger.info("Saved loss plot → %s", out)

    if acc_curves:
        out = results_dir / "plots" / "sft_accuracy.png"
        save_sft_accuracy_plot(acc_curves, out)
        logger.info("Saved accuracy plot → %s", out)

    if failed:
        logger.error("Failed runs: %s", failed)
        sys.exit(1)

    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
