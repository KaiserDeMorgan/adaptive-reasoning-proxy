"""Embedding-based task router (Priority 3 upgrade over the regex classifier).

Embeds the incoming prompt and assigns it to the nearest class *centroid* by
cosine similarity. Centroids are built from a small seed set of labelled
example prompts per task type — expand the seeds (the project plan targets
~500/class) to improve accuracy, or swap in a trained linear head.

This is fully optional and degrades gracefully:
  * If embeddings are disabled or the embedding endpoint is unreachable, it
    falls back to the regex ``classify`` from ``classifier.py``.
  * Centroids are computed lazily on first use and cached.

Configure via env:
  EMBEDDINGS_ENABLED      "1" to enable (default "0" -> always regex fallback)
  EMBEDDING_MODEL         model name (default "text-embedding-3-small")
  EMBEDDING_BASE_URL      OpenAI-compatible /v1 base (defaults to LLM_BASE_URL)
  EMBEDDING_API_KEY       api key (defaults to LLM_API_KEY)
"""

import logging
import os

import numpy as np
from openai import AsyncOpenAI

from classifier import TaskType, classify

logger = logging.getLogger(__name__)

EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "0") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

_embed_client = AsyncOpenAI(
    base_url=os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")),
    api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "none")),
)

# Seed examples per task type. Small on purpose — extend with real labelled data.
SEED_PROMPTS: dict[TaskType, list[str]] = {
    TaskType.FACTUAL: [
        "What is the capital of Japan?",
        "Who wrote Pride and Prejudice?",
        "When did World War II end?",
        "Define photosynthesis.",
        "List the noble gases.",
    ],
    TaskType.REASONING: [
        "Why is the sky blue?",
        "Explain how compound interest works.",
        "Prove that there are infinitely many primes.",
        "Compare TCP and UDP and justify when to use each.",
        "Analyze the causes of the 2008 financial crisis.",
    ],
    TaskType.CODE: [
        "Write a function that checks if a string is a palindrome.",
        "Implement quicksort in Python.",
        "Create a class representing a binary tree.",
        "Write a SQL query to find duplicate rows.",
        "Refactor this loop into a list comprehension.",
    ],
    TaskType.CREATIVE: [
        "Write a short poem about the ocean.",
        "Invent a story about a time-travelling cat.",
        "Describe a sunset on an alien planet.",
        "Compose a haiku about autumn.",
        "Imagine a dialogue between the moon and the sea.",
    ],
}

# Cache: TaskType -> unit-normalised centroid vector.
_centroids: dict[TaskType, np.ndarray] | None = None


async def _embed(texts: list[str]) -> np.ndarray:
    """Return an (n, d) array of embeddings for ``texts``."""
    resp = await _embed_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in resp.data], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-12, None)


async def _ensure_centroids() -> dict[TaskType, np.ndarray]:
    """Build and cache the per-class centroid vectors from the seed prompts."""
    global _centroids
    if _centroids is not None:
        return _centroids

    centroids: dict[TaskType, np.ndarray] = {}
    for task, prompts in SEED_PROMPTS.items():
        vecs = _normalize(await _embed(prompts))
        centroids[task] = _normalize(vecs.mean(axis=0))
    _centroids = centroids
    return _centroids


async def classify_async(prompt: str) -> TaskType:
    """Route ``prompt`` to a :class:`TaskType`.

    Uses embedding + nearest-centroid when enabled and reachable; otherwise
    falls back to the regex classifier. Never raises.
    """
    if not EMBEDDINGS_ENABLED:
        return classify(prompt)

    try:
        centroids = await _ensure_centroids()
        query = _normalize(await _embed([prompt]))[0]
        # Cosine similarity == dot product of unit vectors.
        best_task = max(centroids, key=lambda t: float(query @ centroids[t]))
        return best_task
    except Exception as exc:
        logger.warning("Embedding classify failed, falling back to regex: %s", exc)
        return classify(prompt)
