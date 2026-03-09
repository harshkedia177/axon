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
import threading
from typing import TYPE_CHECKING

from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, NodeLabel
from axon.core.storage.base import NodeEmbedding

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
        # threads=0 lets ONNX Runtime use all CPU cores for intra-op
        # parallelism, which significantly speeds up batch inference.
        model = TextEmbedding(model_name=model_name, threads=0)
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

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def embed_query(query: str, model_name: str = _DEFAULT_MODEL) -> list[float] | None:
    """Embed a single query string, returning ``None`` on failure."""
    if not query or not query.strip():
        return None
    try:
        model = _get_model(model_name)
        return list(next(iter(model.embed([query]))))
    except Exception:
        return None


def _embed_node_list(
    nodes: list[GraphNode],
    graph: KnowledgeGraph,
    model_name: str,
    batch_size: int,
) -> list[NodeEmbedding]:
    """Shared implementation for embedding a list of graph nodes.

    Generates text descriptions, truncates to model context window,
    encodes via fastembed, and returns :class:`NodeEmbedding` objects.
    """
    class_method_idx = build_class_method_index(graph)

    texts: list[str] = []
    valid_nodes: list[GraphNode] = []
    for node in nodes:
        text = generate_text(node, graph, class_method_idx)
        if text and text.strip():
            texts.append(text[:_MAX_TEXT_CHARS])
            valid_nodes.append(node)

    if not texts:
        return []

    logger.info("Embedding %d texts (batch_size=%d) …", len(texts), batch_size)
    model = _get_model(model_name)
    return [
        NodeEmbedding(node_id=node.id, embedding=vector.tolist())
        for node, vector in zip(valid_nodes, model.embed(texts, batch_size=batch_size))
    ]


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes in the graph.

    Uses fastembed's :class:`TextEmbedding` model for batch encoding.
    Each embeddable node is converted to a natural-language description
    via :func:`generate_text`, then embedded in a single batch call.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        model_name: The fastembed model identifier.  Defaults to
            ``"BAAI/bge-small-en-v1.5"``.
        batch_size: Number of texts to encode per batch.  Defaults to 256.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID and its embedding vector as a plain
        Python ``list[float]``.
    """
    all_nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]
    if not all_nodes:
        return []
    return _embed_node_list(all_nodes, graph, model_name, batch_size)


def embed_nodes(
    graph: KnowledgeGraph,
    node_ids: set[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
) -> list[NodeEmbedding]:
    """Like :func:`embed_graph`, but only for the given *node_ids*."""
    if not node_ids:
        return []
    nodes = [graph.get_node(nid) for nid in node_ids]
    nodes = [n for n in nodes if n is not None and n.label in EMBEDDABLE_LABELS]
    if not nodes:
        return []
    return _embed_node_list(nodes, graph, model_name, batch_size)
