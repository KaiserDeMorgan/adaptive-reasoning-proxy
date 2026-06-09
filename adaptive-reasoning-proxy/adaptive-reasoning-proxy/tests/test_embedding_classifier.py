"""Tests for the embedding-based router's fallback behaviour.

We do not hit a real embedding endpoint here; we verify that the router
degrades to the regex classifier when embeddings are disabled or fail.
"""

import pytest

import embedding_classifier as ec
from classifier import TaskType


async def test_falls_back_to_regex_when_disabled(monkeypatch):
    monkeypatch.setattr(ec, "EMBEDDINGS_ENABLED", False)
    assert await ec.classify_async("Write a function to sort a list") == TaskType.CODE
    assert await ec.classify_async("What is the capital of Peru?") == TaskType.FACTUAL


async def test_falls_back_to_regex_when_embedding_errors(monkeypatch):
    monkeypatch.setattr(ec, "EMBEDDINGS_ENABLED", True)

    async def _boom(*args, **kwargs):
        raise RuntimeError("embedding endpoint down")

    monkeypatch.setattr(ec, "_ensure_centroids", _boom)
    # Should swallow the error and route via regex.
    assert await ec.classify_async("Explain why entropy increases") == TaskType.REASONING


async def test_uses_embeddings_when_available(monkeypatch):
    import numpy as np

    monkeypatch.setattr(ec, "EMBEDDINGS_ENABLED", True)
    ec._centroids = None

    # Two orthogonal centroids; query aligns with FACTUAL.
    fake_centroids = {
        TaskType.FACTUAL: np.array([1.0, 0.0]),
        TaskType.CODE: np.array([0.0, 1.0]),
    }

    async def _centroids():
        return fake_centroids

    async def _embed(texts):
        return np.array([[0.9, 0.1]])  # closer to FACTUAL axis

    monkeypatch.setattr(ec, "_ensure_centroids", _centroids)
    monkeypatch.setattr(ec, "_embed", _embed)

    assert await ec.classify_async("anything") == TaskType.FACTUAL
