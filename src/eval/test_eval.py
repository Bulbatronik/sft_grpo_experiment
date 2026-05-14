"""
Evaluate a model checkpoint on the GSM8K test set using vLLM.

Public API
----------
evaluate_model(model_path, test_parquet, ...) -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.reward.gsm8k_reward import compute_score

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    test_parquet: str | Path,
    gpu_memory_utilization: float = 0.85,
    greedy_max_tokens: int = 512,
    sample_n: int = 5,
    sample_temperature: float = 0.7,
    seed: int = 42,
) -> dict:
    """Run greedy and sampled decoding on the full test set.

    Returns a dict with:
        greedy_accuracy, pass_at_1_sampled, per_example (list of dicts)
    """
    try:
        import pandas as pd
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("pandas, vllm, transformers required for evaluate_model") from e

    df = pd.read_parquet(test_parquet)
    prompts = df["prompt"].tolist()
    ground_truths = [row["ground_truth"] for row in df["reward_model"]]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        trust_remote_code=True,
    )

    prompt_strs = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    # Greedy pass.
    greedy_params = SamplingParams(temperature=0.0, max_tokens=greedy_max_tokens, n=1)
    logger.info("Running greedy evaluation on %d examples…", len(prompt_strs))
    greedy_outputs = llm.generate(prompt_strs, greedy_params)

    greedy_correct = []
    for out, gt in zip(greedy_outputs, ground_truths):
        correct = compute_score(out.outputs[0].text, gt) > 0
        greedy_correct.append(int(correct))

    greedy_acc = np.mean(greedy_correct)
    logger.info("Greedy accuracy: %.4f", greedy_acc)

    # Sampled pass.
    sample_params = SamplingParams(
        temperature=sample_temperature, max_tokens=greedy_max_tokens, n=sample_n
    )
    logger.info("Running sampled evaluation (n=%d, T=%.1f)…", sample_n, sample_temperature)
    sample_outputs = llm.generate(prompt_strs, sample_params)

    sampled_pass1 = []
    for out, gt in zip(sample_outputs, ground_truths):
        rewards = [compute_score(o.text, gt) for o in out.outputs]
        sampled_pass1.append(np.mean(rewards))

    per_example = []
    for i, (gc, sp) in enumerate(zip(greedy_correct, sampled_pass1)):
        per_example.append({"index": i, "greedy_correct": gc, "sampled_pass1": sp})

    del llm

    return {
        "greedy_accuracy": float(greedy_acc),
        "pass_at_1_sampled": float(np.mean(sampled_pass1)),
        "per_example": per_example,
    }
