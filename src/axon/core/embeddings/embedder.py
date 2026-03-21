"""Batch embedding pipeline for Axon knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable symbol node, encodes them using *fastembed*, and returns a
list of :class:`NodeEmbedding` objects ready for storage.

Only code-level symbol nodes are embedded.  Structural nodes (Folder,
Community, Process) are deliberately skipped — they lack the semantic
richness that makes embedding worthwhile.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import TYPE_CHECKING

from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.storage.base import EMBEDDING_DIMENSIONS, NodeEmbedding

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class _HttpEmbedder:
    """OpenAI-compatible HTTP embedding client."""

    def __init__(self, base_url: str, model: str, api_key: str = "unused") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

    def _embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        import json
        import urllib.request

        all_vectors: list[list[float]] = []
        expected_dim: int | None = None
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            body = json.dumps({"input": batch, "model": self._model}).encode()
            req = urllib.request.Request(
                f"{self._base_url}/embeddings",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read())

            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError("Embedding response missing 'data' list")
            if len(data) != len(batch):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(data)}"
                )

            indexed_items: list[tuple[int, object]] = []
            for raw_index, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError("Embedding response item must be an object")
                embedding = item.get("embedding")
                index = item.get("index", raw_index)
                if not isinstance(index, int):
                    raise ValueError("Embedding response index must be an integer")
                indexed_items.append((index, embedding))

            indexed_items.sort(key=lambda pair: pair[0])
            actual_indexes = [index for index, _ in indexed_items]
            expected_indexes = list(range(len(batch)))
            if actual_indexes != expected_indexes:
                raise ValueError(
                    f"Embedding response indexes mismatch: expected {expected_indexes}, got {actual_indexes}"
                )

            vectors, expected_dim = _normalize_embedding_batch(
                [embedding for _, embedding in indexed_items],
                expected_count=len(batch),
                expected_dim=expected_dim,
            )
            all_vectors.extend(vectors)
        return all_vectors

    def query_embed(self, text: str):
        return iter(self._embed([text], batch_size=1))

    def passage_embed(self, texts: list[str], batch_size: int = 32):
        return iter(self._embed(texts, batch_size=batch_size))


_model_cache: dict[str, "TextEmbedding | _HttpEmbedder"] = {}
_model_lock = threading.Lock()

_EMBEDDING_BASE_URL = os.environ.get("AXON_EMBEDDING_BASE_URL", "")
_EMBEDDING_MODEL = os.environ.get("AXON_EMBEDDING_MODEL", "")
_EMBEDDING_API_KEY = os.environ.get("AXON_EMBEDDING_API_KEY", "unused")

# BGE-small max sequence is 512 tokens (~2000 chars). Truncating long
# descriptions avoids wasting tokenisation and padding time on text that
# the model would discard anyway.
_MAX_TEXT_CHARS = 2000


def _get_model(model_name: str) -> "TextEmbedding | _HttpEmbedder":
    if _EMBEDDING_BASE_URL and _EMBEDDING_MODEL:
        cache_key = f"http:{_EMBEDDING_BASE_URL}:{_EMBEDDING_MODEL}"
        cached = _model_cache.get(cache_key)
        if cached is not None:
            return cached
        with _model_lock:
            cached = _model_cache.get(cache_key)
            if cached is not None:
                return cached
            model = _HttpEmbedder(_EMBEDDING_BASE_URL, _EMBEDDING_MODEL, _EMBEDDING_API_KEY)
            _model_cache[cache_key] = model
            return model

    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _model_cache.get(model_name)
        if cached is not None:
            return cached
        from fastembed import TextEmbedding

        max_threads = max(2, os.cpu_count() // 2) if os.cpu_count() else 2
        model = TextEmbedding(model_name=model_name, threads=max_threads)
        _model_cache[model_name] = model
        return model


def _get_model_cache_clear() -> None:
    """Clear the model cache (used in tests)."""
    with _model_lock:
        _model_cache.clear()


_get_model.cache_clear = _get_model_cache_clear  # type: ignore[attr-defined]

EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.FILE,
        NodeLabel.FUNCTION,
        NodeLabel.CLASS,
        NodeLabel.METHOD,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
        NodeLabel.ENUM,
    }
)

_DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_DEFAULT_DIMENSIONS = EMBEDDING_DIMENSIONS
_DEFAULT_BATCH_SIZE = 32
_MAX_TEXT_CHARS = 8192


def _normalize_embedding_vector(vector: object, *, dimensions: int | None = None) -> list[float]:
    raw_values = getattr(vector, "tolist")() if hasattr(vector, "tolist") else vector
    if isinstance(raw_values, (str, bytes)):
        raise ValueError("Embedding vector must be a sequence of numeric values")

    try:
        values = list(raw_values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Embedding vector must be iterable") from exc

    if not values:
        raise ValueError("Embedding vector must not be empty")

    normalized: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Embedding vector contains non-finite value: {number}")
        normalized.append(number)

    if dimensions is not None:
        normalized = normalized[:dimensions]
        if len(normalized) != dimensions:
            raise ValueError(
                f"Expected embedding of {dimensions} dimensions, got {len(normalized)}"
            )

    return normalized


def _normalize_embedding_batch(
    vectors: list[object],
    *,
    expected_count: int,
    expected_dim: int | None = None,
    dimensions: int | None = None,
) -> tuple[list[list[float]], int]:
    if len(vectors) != expected_count:
        raise ValueError(f"Embedding count mismatch: expected {expected_count}, got {len(vectors)}")

    normalized_vectors: list[list[float]] = []
    dimension = expected_dim
    for vector in vectors:
        normalized = _normalize_embedding_vector(vector, dimensions=dimensions)
        if dimension is None:
            dimension = len(normalized)
        elif len(normalized) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {dimension}, got {len(normalized)}"
            )
        normalized_vectors.append(normalized)

    if dimension is None:
        raise ValueError("Expected at least one embedding vector")

    return normalized_vectors, dimension


def embed_query(
    query: str,
    model_name: str = _DEFAULT_MODEL,
    dimensions: int = _DEFAULT_DIMENSIONS,
) -> list[float] | None:
    if not query or not query.strip():
        return None
    try:
        model = _get_model(model_name)
        vectors, _ = _normalize_embedding_batch(
            list(model.query_embed(query)),
            expected_count=1,
            dimensions=dimensions,
        )
        return vectors[0]
    except Exception:
        logger.warning("embed_query failed", exc_info=True)
        return None


def _embed_node_list(
    nodes: list,
    texts: list[str],
    model_name: str,
    batch_size: int,
    dimensions: int,
) -> list[NodeEmbedding]:
    if not texts:
        return []

    model = _get_model(model_name)
    vectors, _ = _normalize_embedding_batch(
        list(model.passage_embed(texts, batch_size=batch_size)),
        expected_count=len(nodes),
        dimensions=dimensions,
    )

    results: list[NodeEmbedding] = []
    for node, vector in zip(nodes, vectors):
        results.append(NodeEmbedding(node_id=node.id, embedding=vector))
    return results


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dimensions: int = _DEFAULT_DIMENSIONS,
) -> list[NodeEmbedding]:
    all_nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]
    if not all_nodes:
        return []

    class_method_idx = build_class_method_index(graph)

    texts: list[str] = []
    nodes = []
    for node in all_nodes:
        text = generate_text(node, graph, class_method_idx)
        if text and text.strip():
            texts.append(text[:_MAX_TEXT_CHARS])
            nodes.append(node)

    if not texts:
        return []

    return _embed_node_list(nodes, texts, model_name, batch_size, dimensions)


def embed_nodes(
    graph: KnowledgeGraph,
    node_ids: set[str],
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dimensions: int = _DEFAULT_DIMENSIONS,
) -> list[NodeEmbedding]:
    if not node_ids:
        return []
    nodes = [graph.get_node(nid) for nid in node_ids]
    nodes = [n for n in nodes if n is not None and n.label in EMBEDDABLE_LABELS]
    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)

    texts: list[str] = []
    valid_nodes = []
    for node in nodes:
        text = generate_text(node, graph, class_method_idx)
        if text and text.strip():
            texts.append(text[:_MAX_TEXT_CHARS])
            valid_nodes.append(node)

    if not texts:
        return []

    return _embed_node_list(valid_nodes, texts, model_name, batch_size, dimensions)
