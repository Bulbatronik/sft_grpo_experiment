"""
GRPO subset selection strategies.

Public API
----------
variance_select(stats, budget, seed)  -> np.ndarray of indices
random_select(n_total, budget, seed)  -> np.ndarray of indices
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def variance_select(
    std_rewards: np.ndarray,
    mean_rewards: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Select examples with the highest reward std (best GRPO signal).

    Tiebreaker: Bernoulli variance peaks at mean=0.5, so among equal-std
    examples prefer those whose mean is closest to 0.5.

    Args:
        std_rewards:  (N,) per-example reward standard deviation over rollouts.
        mean_rewards: (N,) per-example mean reward over rollouts.
        budget:       number of examples to select.

    Returns:
        1-D integer array of selected indices, length budget.
    """
    n = len(std_rewards)
    budget = min(budget, n)

    # Tiebreaker: closeness of mean to 0.5 → higher is better → negate distance.
    tiebreak = -np.abs(mean_rewards - 0.5)

    # Lexicographic sort: primary = std (desc), secondary = tiebreak (desc).
    order = np.lexsort((tiebreak, std_rewards))[::-1]
    selected = order[:budget]
    logger.info(
        "Variance select: top-%d  std range [%.3f, %.3f]  mean range [%.3f, %.3f]",
        budget,
        std_rewards[selected].min(),
        std_rewards[selected].max(),
        mean_rewards[selected].min(),
        mean_rewards[selected].max(),
    )
    return selected.astype(np.int64)


def random_select(
    n_total: int,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Uniform random sample without replacement."""
    rng = np.random.default_rng(seed)
    budget = min(budget, n_total)
    return rng.choice(n_total, size=budget, replace=False)
