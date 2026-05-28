"""
Rollout-based scoring of a candidate pool using vLLM (inference-only, no gradients).

Public API
----------
score_pool(model_path, prompts, ground_truths, n_rollouts, ...) -> list[dict]
"""

from __future__ import annotations

import gc
import logging

from tqdm import tqdm

from src.reward.gsm8k_reward import compute_score
from src.utils.checkpoint import is_lora_checkpoint, read_adapter_config

logger = logging.getLogger(__name__)


def score_pool(
    model_path: str,
    prompts: list[list[dict]],
    ground_truths: list[str],
    n_rollouts: int = 5,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    gpu_memory_utilization: float = 0.85,
    seed: int = 42,
) -> list[dict]:
    """Score each prompt with n_rollouts generations; return per-example stats.

    Accepts either a full HF model checkpoint or a LoRA adapter directory
    (detected by the presence of adapter_config.json). LoRA adapters are loaded
    via vLLM's native LoRA support — no merging or temp files needed.

    Returns a list of dicts with keys:
        rewards, mean_reward, std_reward, pass_rate, pass_at_k
    """
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("vllm and transformers are required for score_pool") from e

    lora_request = None
    if is_lora_checkpoint(model_path):
        from vllm.lora.request import LoRARequest
        adapter_cfg = read_adapter_config(model_path)
        base_model = adapter_cfg["base_model_name_or_path"]
        logger.info("LoRA checkpoint detected. Base model: %s  Adapter: %s", base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(
            model=base_model,
            enable_lora=True,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            trust_remote_code=True,
        )
        lora_request = LoRARequest("sft_adapter", 1, model_path)
    else:
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
    generate_kwargs = {"lora_request": lora_request} if lora_request else {}
    outputs = llm.generate(prompt_strs, sampling_params, **generate_kwargs)

    results = []
    for output, gt in tqdm(zip(outputs, ground_truths), total=len(outputs), desc="scoring"):
        rewards = [compute_score(o.text, gt) for o in output.outputs]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        pass_at_k = int(any(r > 0 for r in rewards))
        results.append(
            {
                "rewards": rewards,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "pass_rate": mean_r,
                "pass_at_k": pass_at_k,
            }
        )

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return results
