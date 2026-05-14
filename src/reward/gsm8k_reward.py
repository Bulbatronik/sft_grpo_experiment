"""
GSM8K reward function – single source of truth for both standalone rollout
scoring (Phase 3) and verl GRPO training.

For GRPO training, verl routes to its built-in gsm8k scorer via
data_source=="openai/gsm8k" in default_compute_score, which is identical
to the logic here.  This file is the standalone copy used by score_pool.py.
"""

import re

_CLIP = 300  # only search the last N chars – fast and sufficient


def extract_answer(solution_str: str) -> str | None:
    """Return the last #### <number> token, or None if absent."""
    tail = solution_str[-_CLIP:] if len(solution_str) > _CLIP else solution_str
    matches = re.findall(r"#### (\-?[0-9\.,]+)", tail)
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "")


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if extracted answer matches ground truth, else 0.0."""
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth else 0.0
