"""Hybrid search combining full-text and vector search via Reciprocal Rank Fusion.

Reciprocal Rank Fusion (RRF) merges ranked lists from different retrieval
systems into a single ranking.  Each document receives a score::

    RRF_score(d) = sum_r  weight_r / (k + rank_r(d))

where *k* is a smoothing constant (default 60) that prevents high-ranked items
from dominating, *rank_r(d)* is the 1-based position of document *d* in ranker
*r*'s result list, and *weight_r* scales that ranker's contribution.
"""

from __future__ import annotations

from dataclasses import replace

from axon.core.storage.base import SearchResult, StorageBackend


def hybrid_search(
    query: str,
    storage: StorageBackend,
    query_embedding: list[float] | None = None,
    limit: int = 20,
    fts_weight: float = 1.0,
    vector_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Run hybrid search combining FTS and vector search with RRF.

    Parameters:
        query: The text query for keyword search.
        storage: The storage backend to search against.
        query_embedding: Pre-computed query embedding vector.
            If ``None``, only full-text search is used.
        limit: Maximum number of results to return.
        fts_weight: Weight multiplier for FTS results in RRF scoring.
        vector_weight: Weight multiplier for vector results in RRF scoring.
        rrf_k: RRF smoothing constant (standard value is 60).

    Returns:
        Merged list of :class:`SearchResult` sorted by combined RRF score,
        highest first.
    """
    if limit <= 0:
        return []

    candidate_limit = limit * 3

    # Step 1: gather ranked lists from each source
    fts_results = storage.fts_search(query, limit=candidate_limit)

    # Fuzzy fallback: if BM25 returns nothing, try Levenshtein name matching.
    if not fts_results and hasattr(storage, "fuzzy_search"):
        fts_results = storage.fuzzy_search(query, limit=candidate_limit)

    vector_results: list[SearchResult] = []
    if query_embedding is not None:
        vector_results = storage.vector_search(query_embedding, limit=candidate_limit)

    rrf_scores: dict[str, float] = {}
    metadata: dict[str, SearchResult] = {}

    _accumulate_ranks(fts_results, fts_weight, rrf_k, rrf_scores, metadata)
    _accumulate_ranks(vector_results, vector_weight, rrf_k, rrf_scores, metadata)

    merged: list[SearchResult] = []
    for node_id, score in rrf_scores.items():
        source = metadata[node_id]
        merged.append(
            replace(source, score=score),
        )

    merged.sort(key=lambda r: r.score, reverse=True)

    return merged[:limit]


def _accumulate_ranks(
    results: list[SearchResult],
    weight: float,
    k: int,
    scores: dict[str, float],
    metadata: dict[str, SearchResult],
) -> None:
    """Add RRF contributions from a single ranked list.

    Only the first occurrence of each ``node_id`` in *results* is considered
    (i.e. duplicates within the same list are ignored).
    """
    seen: set[str] = set()
    for rank_0, result in enumerate(results):
        nid = result.node_id
        if nid in seen:
            continue
        seen.add(nid)

        rank_1 = rank_0 + 1  # 1-based rank
        scores[nid] = scores.get(nid, 0.0) + weight / (k + rank_1)

        # Keep the first metadata we encounter for this node
        if nid not in metadata:
            metadata[nid] = result
