"""
SFT subset selection strategies.

Each strategy is a function that maps a pool of N training examples to a
1-D integer array of `budget` selected indices. Strategies are registered in
STRATEGIES so the selection script can dispatch by name; adding a new strategy
means writing one function and adding one registry entry.

Strategies and the inputs they need:
    random     — nothing beyond the pool size.
    diverse    — PCA-reduced embeddings (farthest-point sampling).
    ops        — GSM8K solution strings (counts intermediate calculations).
    sentences  — GSM8K solution strings (counts reasoning sentences).
    ifd_hard   — IFD scores (highest = question helps least = hardest).
    ifd_easy   — IFD scores (lowest = easiest).
    ifd_mid    — IFD scores (band around the median).

Public API
----------
select(name, n_total, budget, seed, *,
       reduced=None, solutions=None, ifd_scores=None) -> np.ndarray
STRATEGIES — mapping of strategy name → metadata (what inputs it requires)
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


# ── Strategy implementations ──────────────────────────────────────────────────

def random_select(n_total: int, budget: int, seed: int = 42) -> np.ndarray:
    """Uniform random sample without replacement (the baseline)."""
    rng = np.random.default_rng(seed)
    budget = min(budget, n_total)
    return rng.choice(n_total, size=budget, replace=False)


def diverse_select(reduced: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    """Farthest-point sampling (FPS) in PCA-reduced embedding space.

    What FPS does
    -------------
    FPS greedily builds a subset that is maximally spread out. Starting from a
    seed point, it repeatedly adds the example whose distance to its *nearest
    already-selected* example is largest:

        1. Pick a seed point (here: the most "extreme" example — the one with
           the largest summed |z| score across components — so the walk starts
           at the boundary of the distribution rather than its centre).
        2. For every candidate, track d(i) = distance to the closest selected
           point so far.
        3. Add argmax_i d(i) to the subset; update all d(i) with distances to
           the new point (an O(N) update, so the whole run is O(N · budget)).
        4. Repeat until `budget` points are selected.

    The result approximates a maximin / k-center design: no region of the
    embedding space that contains data is left without a selected example, so
    rare question types are guaranteed representation. The trade-off is that
    FPS is attracted to outliers — the very first picks are the most unusual
    examples in the pool (this is also what makes it a *diversity* method
    rather than a *density* method).

    Distances are computed after standardising each PCA component to unit
    variance, so low-variance (later) components contribute as much as the
    dominant ones; without this, FPS would effectively only see PC1/PC2.

    FPS is deterministic given the seed point, so re-runs are reproducible
    without relying on the RNG.

    Args:
        reduced: (N, K) PCA-projected embeddings (zero-centred by PCA).
        budget:  number of examples to select.
        seed:    kept for interface symmetry; FPS itself is deterministic.

    Returns:
        1-D integer array of selected indices, length budget.
    """
    n = reduced.shape[0]
    budget = min(budget, n)

    # Standardise each component to unit variance so all axes contribute equally.
    stds = reduced.std(axis=0)
    stds[stds == 0] = 1.0
    z = reduced / stds  # (N, K)

    # Seed: the most extreme point (largest summed |z|), i.e. start at the
    # boundary of the distribution.
    z_scores = np.abs(z).sum(axis=1)
    seed_idx = int(np.argmax(z_scores))
    logger.info("FPS seed point index=%d  |z|_sum=%.3f", seed_idx, z_scores[seed_idx])

    selected = [seed_idx]
    # min_dists[i] = squared distance from point i to its nearest selected point.
    min_dists = np.sum((z - z[seed_idx]) ** 2, axis=1)

    for step in range(budget - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        new_dists = np.sum((z - z[next_idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)

        if (step + 1) % 100 == 0:
            logger.debug("FPS step %d/%d", step + 1, budget - 1)

    return np.array(selected, dtype=np.int64)


# GSM8K ground-truth solutions annotate every intermediate calculation as
# <<expression=result>>, e.g. "She has <<2+3=5>>5 apples." — counting these
# markers counts the arithmetic steps the solution requires.
_CALC_RE = re.compile(r"<<")
_SENT_RE = re.compile(r"[.!?\n]+")


def _n_operations(solution: str) -> int:
    return len(_CALC_RE.findall(solution))


def _n_sentences(solution: str) -> int:
    # Strip the final "#### N" line first — it is not a reasoning step.
    body = solution.split("####")[0]
    return len([s for s in _SENT_RE.split(body) if s.strip()])


def complexity_select(
    solutions: list[str],
    budget: int,
    proxy: str = "ops",
    seed: int = 42,
) -> np.ndarray:
    """Select the `budget` most complex examples by a reasoning-complexity proxy.

    Proxies (both computed from the ground-truth solution text, so they cost
    nothing — no model forward pass needed):
        ops       — number of <<...>> intermediate calculations (arithmetic steps).
        sentences — number of sentences in the reasoning before the #### answer.

    Ties are broken randomly (seeded) so the selection isn't biased by pool
    order when many examples share the same score.
    """
    scorer = {"ops": _n_operations, "sentences": _n_sentences}[proxy]
    scores = np.array([scorer(s) for s in solutions], dtype=np.float64)
    logger.info(
        "Complexity proxy %s: min=%d  median=%d  max=%d",
        proxy, int(scores.min()), int(np.median(scores)), int(scores.max()),
    )

    rng = np.random.default_rng(seed)
    tiebreak = rng.random(len(scores))
    order = np.lexsort((tiebreak, -scores))  # descending score, random ties
    budget = min(budget, len(scores))
    return order[:budget].astype(np.int64)


def ifd_select(
    scores: np.ndarray,
    budget: int,
    mode: str = "hard",
    seed: int = 42,
) -> np.ndarray:
    """Select by Instruction-Following Difficulty score (see src/data/ifd.py).

    IFD = CE(solution | question) / CE(solution): how little the question
    helps the model predict the solution. Modes:
        hard — highest IFD first (the paper-style "hard examples" pick).
        easy — lowest IFD first.
        mid  — the `budget` examples closest to the median IFD (the middle
               band of the difficulty distribution).

    Ties are broken randomly (seeded).
    """
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(len(scores))
    if mode == "hard":
        key = -scores
    elif mode == "easy":
        key = scores
    elif mode == "mid":
        key = np.abs(scores - np.median(scores))
    else:
        raise ValueError(f"Unknown IFD mode {mode!r}; use hard, easy, or mid.")
    order = np.lexsort((tiebreak, key))
    budget = min(budget, len(scores))
    return order[:budget].astype(np.int64)


# ── Registry and dispatcher ───────────────────────────────────────────────────

# name → dict of requirements; the selection script uses `needs` to decide
# whether to compute embeddings / load solution strings.
STRATEGIES: dict[str, dict] = {
    "random":    {"needs": set()},
    "diverse":   {"needs": {"reduced"}},
    "ops":       {"needs": {"solutions"}},
    "sentences": {"needs": {"solutions"}},
    "ifd_hard":  {"needs": {"ifd_scores"}},
    "ifd_easy":  {"needs": {"ifd_scores"}},
    "ifd_mid":   {"needs": {"ifd_scores"}},
}


def select(
    name: str,
    n_total: int,
    budget: int,
    seed: int = 42,
    *,
    reduced: np.ndarray | None = None,
    solutions: list[str] | None = None,
    ifd_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to a selection strategy by registry name."""
    if name == "random":
        return random_select(n_total, budget, seed)
    if name == "diverse":
        assert reduced is not None, "diverse strategy needs PCA-reduced embeddings"
        return diverse_select(reduced, budget, seed)
    if name in ("ops", "sentences"):
        assert solutions is not None, f"{name} strategy needs solution strings"
        return complexity_select(solutions, budget, proxy=name, seed=seed)
    if name.startswith("ifd_"):
        assert ifd_scores is not None, "IFD strategies need precomputed scores"
        return ifd_select(ifd_scores, budget, mode=name.removeprefix("ifd_"), seed=seed)
    raise ValueError(f"Unknown strategy {name!r}; available: {sorted(STRATEGIES)}")
