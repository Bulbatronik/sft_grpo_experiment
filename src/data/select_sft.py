"""
SFT subset selection strategies.

Public API
----------
diverse_select(reduced, budget, seed)  -> np.ndarray of indices
random_select(n_total, budget, seed)   -> np.ndarray of indices
quantile_uniform_select(reduced, budget, n_bins, seed) -> np.ndarray of indices
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def diverse_select(
    reduced: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Farthest-point sampling seeded by the point with the highest |z|-sum.

    Args:
        reduced: (N, K) PCA-projected embeddings (zero-centred by PCA).
        budget:  number of examples to select.
        seed:    unused beyond reproducibility note; FPS is deterministic once
                 the seed point is fixed.

    Returns:
        1-D integer array of selected indices, length budget.
    """
    rng = np.random.default_rng(seed)
    n = reduced.shape[0]
    budget = min(budget, n)

    # Standardise each component to unit variance so all axes contribute equally.
    stds = reduced.std(axis=0)
    stds[stds == 0] = 1.0
    z = reduced / stds  # (N, K)

    # Seed: pick the point with the highest sum of |z| scores.
    z_scores = np.abs(z).sum(axis=1)
    seed_idx = int(np.argmax(z_scores))
    logger.info("FPS seed point index=%d  |z|_sum=%.3f", seed_idx, z_scores[seed_idx])

    selected = [seed_idx]
    # min-dist array: dist from each point to its nearest selected point
    min_dists = np.full(n, np.inf)
    seed_vec = z[seed_idx]
    min_dists = np.sum((z - seed_vec) ** 2, axis=1)

    for step in range(budget - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        # Update min distances with the newly added point.
        new_dists = np.sum((z - z[next_idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)

        if (step + 1) % 100 == 0:
            logger.debug("FPS step %d/%d", step + 1, budget - 1)

    return np.array(selected, dtype=np.int64)


def random_select(
    n_total: int,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Uniform random sample without replacement."""
    rng = np.random.default_rng(seed)
    budget = min(budget, n_total)
    return rng.choice(n_total, size=budget, replace=False)


def quantile_uniform_select(
    reduced: np.ndarray,
    budget: int,
    n_bins: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Bin each PCA axis into n_bins quantile bins; sample uniformly across populated bin keys.

    This is the --diversity-method quantile-uniform alternative to FPS.
    """
    rng = np.random.default_rng(seed)
    n, k = reduced.shape
    budget = min(budget, n)

    # Assign each example a tuple bin key across all retained components.
    bin_keys = np.zeros((n, k), dtype=np.int32)
    for j in range(k):
        col = reduced[:, j]
        quantiles = np.percentile(col, np.linspace(0, 100, n_bins + 1))
        quantiles = np.unique(quantiles)
        bin_keys[:, j] = np.searchsorted(quantiles[1:-1], col, side="right")

    # Group indices by bin key tuple.
    from collections import defaultdict
    bins: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n):
        bins[tuple(bin_keys[i].tolist())].append(i)

    bin_list = list(bins.values())
    rng.shuffle(bin_list)

    selected: list[int] = []
    # Round-robin across bins until budget filled.
    bin_ptrs = [0] * len(bin_list)
    for b in bin_list:
        rng.shuffle(b)

    i = 0
    while len(selected) < budget:
        bin_idx = i % len(bin_list)
        ptr = bin_ptrs[bin_idx]
        if ptr < len(bin_list[bin_idx]):
            selected.append(bin_list[bin_idx][ptr])
            bin_ptrs[bin_idx] += 1
        i += 1
        if all(bin_ptrs[b] >= len(bin_list[b]) for b in range(len(bin_list))):
            break

    return np.array(selected[:budget], dtype=np.int64)
