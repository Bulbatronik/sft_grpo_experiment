"""
Convert the HuggingFace openai/gsm8k dataset into verl-compatible parquet files.

Public API
----------
build_rows(split_data) -> list[dict]
extract_final_number(answer_str) -> str
"""

from __future__ import annotations

import re
from typing import Any

_SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step and show your reasoning. "
    "At the end of your response, write your final answer in the format: #### <number>"
)

_FINAL_RE = re.compile(r"####\s*(\-?[0-9\.,]+)")


def extract_final_number(answer_str: str) -> str:
    """Extract the numeric answer after #### from a GSM8K ground-truth string."""
    matches = _FINAL_RE.findall(answer_str)
    if not matches:
        raise ValueError(f"No #### answer found in: {answer_str!r}")
    return matches[-1].replace(",", "").replace("$", "").strip()


def build_rows(split_data: Any, split_offset: int = 0) -> list[dict]:
    """Convert a HuggingFace dataset split into verl-format row dicts.

    Columns produced:
      prompt    — [system, user] messages for GRPO (no assistant turn).
      messages  — [system, user, assistant] messages for SFT (full conversation).
      response  — raw ground-truth string (kept for reference).
      data_source, reward_model, extra_info — standard verl fields.
    """
    rows = []
    for i, example in enumerate(split_data):
        question: str = example["question"]
        full_answer: str = example["answer"]
        final_number = extract_final_number(full_answer)

        prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        messages = prompt + [{"role": "assistant", "content": full_answer}]

        rows.append(
            {
                "prompt": prompt,
                "messages": messages,
                "response": full_answer,
                "data_source": "gsm8k",
                "reward_model": {"style": "rule", "ground_truth": final_number},
                "extra_info": {
                    "index": split_offset + i,
                    "question": question,
                    "answer": full_answer,
                },
            }
        )
    return rows
