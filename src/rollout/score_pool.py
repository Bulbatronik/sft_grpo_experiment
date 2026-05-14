"""
Rollout-based scoring of a candidate pool using vLLM (inference-only, no gradients).

Public API
----------
score_pool(model_path, prompts, ground_truths, n_rollouts, ...) -> list[dict]
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from src.reward.gsm8k_reward import compute_score

logger = logging.getLogger(__name__)


def score_pool(
    model_path: str,
    prompts: list[list[dict]],
    ground_truths: list[str],
    n_rollouts: int = 5,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    gpu_memory_utilization: float = 0.85,
    seed: int = 42,
) -> list[dict]:
    """Score each prompt with n_rollouts generations; return per-example stats.

    Returns a list of dicts with keys:
        rewards, mean_reward, std_reward, pass_at_1, pass_at_k
    """
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("vllm and transformers are required for score_pool") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=n_rollouts,
    )

    # Apply chat template to get flat strings.
    prompt_strs = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    logger.info("Running rollouts for %d prompts (n=%d each)…", len(prompt_strs), n_rollouts)
    outputs = llm.generate(prompt_strs, sampling_params)

    results = []
    for output, gt in tqdm(zip(outputs, ground_truths), total=len(outputs), desc="scoring"):
        rewards = [
            compute_score(o.text, gt) for o in output.outputs
        ]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        pass_at_k = 1.0 if any(r > 0 for r in rewards) else 0.0
        results.append(
            {
                "rewards": rewards,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "pass_at_1": mean_r,
                "pass_at_k": pass_at_k,
            }
        )

    del llm
    return results
