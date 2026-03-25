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

_model_cache: dict[str, "TextEmbedding"] = {}
_model_lock = threading.Lock()

# BGE-small max sequence is 512 tokens (~2000 chars).  Truncating long
# descriptions avoids wasting tokenisation and padding time on text that
# the model would discard anyway.
_MAX_TEXT_CHARS = 2000


def _get_model(model_name: str) -> "TextEmbedding":
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _model_cache.get(model_name)
        if cached is not None:
            return cached
        from fastembed import TextEmbedding

        # Cap ONNX threads to avoid saturating all CPU cores.
        # Default to half the available cores (minimum 2).
        max_threads = max(2, os.cpu_count() // 2) if os.cpu_count() else 2
        model = TextEmbedding(model_name=model_name, threads=max_threads)
        _model_cache[model_name] = model
        return model


def _get_model_cache_clear() -> None:
    """Clear the model cache (used in tests)."""
    with _model_lock:
        _model_cache.clear()


_get_model.cache_clear = _get_model_cache_clear  # type: ignore[attr-defined]

# Labels worth embedding — skip Folder, Community, Process (structural only).
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
_DEFAULT_DIMENSIONS = EMBEDDING_DIMENSIONS  # 384 via Matryoshka
_DEFAULT_BATCH_SIZE = 32
_MAX_TEXT_CHARS = 8192


def embed_query(
    query: str,
    model_name: str = _DEFAULT_MODEL,
    dimensions: int = _DEFAULT_DIMENSIONS,
) -> list[float] | None:
    """Embed a single query string, returning ``None`` on failure."""
    if not query or not query.strip():
        return None
    try:
        model = _get_model(model_name)
        vec = next(iter(model.query_embed(query)))
        return vec[:dimensions].tolist()
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
    """Embed a list of nodes with their corresponding texts."""
    if not texts:
        return []

    model = _get_model(model_name)
    vectors = list(model.passage_embed(texts, batch_size=batch_size))

    results: list[NodeEmbedding] = []
    for node, vector in zip(nodes, vectors):
        results.append(
            NodeEmbedding(
                node_id=node.id,
                embedding=vector[:dimensions].tolist(),
            )
        )

    return results


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dimensions: int = _DEFAULT_DIMENSIONS,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes in the graph.

    Uses fastembed's :class:`TextEmbedding` model for batch encoding.
    Each embeddable node is converted to a natural-language description
    via :func:`generate_text`, then embedded in a single batch call.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        model_name: The fastembed model identifier.  Defaults to
            ``"nomic-ai/nomic-embed-text-v1.5"``.
        batch_size: Number of texts to encode per batch.  Defaults to 128.
        dimensions: Number of dimensions for Matryoshka truncation.
            Defaults to 384.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID and its embedding vector as a plain
        Python ``list[float]``.
    """
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
    """Like :func:`embed_graph`, but only for the given *node_ids*."""
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
