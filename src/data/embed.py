"""
Sentence-level embeddings + PCA for the GSM8K training pool.

Public API
----------
embed_questions(questions, model_name, cache_path, force)  -> np.ndarray (N, D)
fit_pca(embeddings, variance_threshold)                     -> (PCA, np.ndarray)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger(__name__)


def embed_questions(
    questions: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_path: str | Path | None = None,
    force: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    """Return (N, D) float32 array of sentence embeddings."""
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists() and not force:
        logger.info("Loading cached embeddings from %s", cache_path)
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Embedding %d questions in batches of %d", len(questions), batch_size)
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Saved embeddings to %s", cache_path)

    return embeddings.astype(np.float32)


def fit_pca(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA; keep the smallest number of components whose cumulative
    explained variance reaches `variance_threshold`. No component cap — the
    threshold alone decides the dimensionality.

    Returns (fitted PCA, transformed embeddings of shape (N, n_keep)).
    """
    # Fit the full decomposition once to read the variance spectrum.
    n_max = min(embeddings.shape[1], embeddings.shape[0] - 1)
    full_pca = PCA(n_components=n_max, random_state=42)
    full_pca.fit(embeddings)

    cumvar = np.cumsum(full_pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_keep = min(n_keep, n_max)

    logger.info(
        "PCA: keeping %d / %d components (%.1f%% variance explained, threshold %.0f%%)",
        n_keep,
        n_max,
        cumvar[n_keep - 1] * 100,
        variance_threshold * 100,
    )

    pca = PCA(n_components=n_keep, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pca, reduced.astype(np.float32)
