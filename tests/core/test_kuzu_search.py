"""Tests for KuzuDB FTS search, embedding storage, and vector search."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.graph.model import GraphNode, NodeLabel, generate_id
from axon.core.storage.base import NodeEmbedding
from axon.core.storage.kuzu_backend import KuzuBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> KuzuBackend:
    """Return a KuzuBackend initialised in a temporary directory."""
    db_path = tmp_path / "search_test_db"
    b = KuzuBackend()
    b.initialize(db_path)
    yield b
    b.close()


def _make_node(
    label: NodeLabel = NodeLabel.FUNCTION,
    file_path: str = "src/app.py",
    name: str = "my_func",
    content: str = "",
    signature: str = "",
) -> GraphNode:
    """Helper to build a GraphNode with a deterministic id."""
    return GraphNode(
        id=generate_id(label, file_path, name),
        label=label,
        name=name,
        file_path=file_path,
        content=content,
        signature=signature,
    )


# ---------------------------------------------------------------------------
# FTS search tests
# ---------------------------------------------------------------------------


class TestFtsSearch:
    """Full-text search across node tables."""

    def test_exact_name_match(self, backend: KuzuBackend) -> None:
        """Searching a name that exists should return a result with a positive BM25 score."""
        node = _make_node(name="process_data", content="does stuff")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("process_data", limit=10)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score > 0
        assert top.node_name == "process_data"

    def test_partial_name_match(self, backend: KuzuBackend) -> None:
        """A query matching part of the name should still find the node."""
        node = _make_node(name="process_data_pipeline", content="")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("process_data", limit=10)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score > 0

    def test_content_match(self, backend: KuzuBackend) -> None:
        """A query found in content should match via BM25."""
        node = _make_node(name="unrelated_name", content="this calls process_data inside")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("process_data", limit=10)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score > 0

    def test_no_match(self, backend: KuzuBackend) -> None:
        """When no nodes match, return an empty list."""
        node = _make_node(name="hello", content="world")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("nonexistent_symbol", limit=10)
        assert results == []

    def test_limit_respected(self, backend: KuzuBackend) -> None:
        """Only *limit* results should be returned."""
        nodes = [
            _make_node(name=f"func_{i}", file_path=f"src/f{i}.py", content="common_term")
            for i in range(5)
        ]
        backend.add_nodes(nodes)
        backend.rebuild_fts_indexes()

        results = backend.fts_search("common_term", limit=3)
        assert len(results) == 3

    def test_case_insensitive(self, backend: KuzuBackend) -> None:
        """BM25 search should handle case differences."""
        node = _make_node(name="ProcessData", content="")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("processdata", limit=10)
        assert len(results) >= 1
        assert results[0].node_id == node.id

    def test_score_ordering(self, backend: KuzuBackend) -> None:
        """Nodes with the query term in the name should rank above content-only matches."""
        name_match = _make_node(name="target", file_path="src/a.py", content="")
        content_only = _make_node(
            name="unrelated", file_path="src/c.py", content="has target in body"
        )
        backend.add_nodes([name_match, content_only])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("target", limit=10)
        assert len(results) >= 2
        # Name match should score higher than content-only
        assert results[0].node_id == name_match.id
        assert results[0].score >= results[1].score

    def test_result_fields_populated(self, backend: KuzuBackend) -> None:
        """SearchResult should have node_name, file_path, label, and snippet."""
        node = _make_node(
            label=NodeLabel.CLASS,
            name="MyClass",
            file_path="src/models.py",
            content="class body here",
        )
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("MyClass", limit=10)
        assert len(results) >= 1
        r = results[0]
        assert r.node_name == "MyClass"
        assert r.file_path == "src/models.py"
        assert r.label == "class"
        assert r.snippet != ""

    def test_signature_match(self, backend: KuzuBackend) -> None:
        """A query found in the signature field should also match via BM25."""
        node = _make_node(
            name="unrelated",
            content="",
            signature="def special_function(x: int) -> str",
        )
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("special_function", limit=10)
        assert len(results) >= 1
        assert results[0].node_id == node.id
        assert results[0].score > 0


# ---------------------------------------------------------------------------
# Embedding storage and vector search tests
# ---------------------------------------------------------------------------


class TestEmbeddingsAndVectorSearch:
    """store_embeddings + vector_search round-trip tests."""

    def test_store_and_retrieve_by_vector(self, backend: KuzuBackend) -> None:
        """Stored embeddings should be retrievable via vector_search."""
        # Insert a node so we can populate SearchResult fields.
        node = _make_node(name="embed_func", file_path="src/embed.py", content="body")
        backend.add_nodes([node])

        # Store an embedding for that node.
        emb = NodeEmbedding(node_id=node.id, embedding=[1.0, 0.0, 0.0])
        backend.store_embeddings([emb])

        # Search with the same vector -- cosine similarity should be 1.0.
        results = backend.vector_search([1.0, 0.0, 0.0], limit=5)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score == pytest.approx(1.0, abs=1e-6)
        assert top.node_name == "embed_func"

    def test_vector_search_empty(self, backend: KuzuBackend) -> None:
        """When no embeddings exist, vector_search returns an empty list."""
        results = backend.vector_search([1.0, 0.0, 0.0], limit=5)
        assert results == []

    def test_vector_search_ranking(self, backend: KuzuBackend) -> None:
        """Closer vectors should rank higher."""
        n1 = _make_node(name="close_func", file_path="src/a.py")
        n2 = _make_node(name="far_func", file_path="src/b.py")
        backend.add_nodes([n1, n2])

        # close_func embedding is close to query, far_func is orthogonal.
        backend.store_embeddings(
            [
                NodeEmbedding(node_id=n1.id, embedding=[0.9, 0.1, 0.0]),
                NodeEmbedding(node_id=n2.id, embedding=[0.0, 0.0, 1.0]),
            ]
        )

        results = backend.vector_search([1.0, 0.0, 0.0], limit=5)
        assert len(results) == 2
        assert results[0].node_id == n1.id
        assert results[0].score > results[1].score

    def test_vector_search_limit(self, backend: KuzuBackend) -> None:
        """Only *limit* results should be returned from vector_search."""
        nodes = []
        embeddings = []
        for i in range(5):
            n = _make_node(name=f"vfunc_{i}", file_path=f"src/v{i}.py")
            nodes.append(n)
            # All somewhat similar embeddings.
            vec = [0.0] * 5
            vec[i] = 1.0
            embeddings.append(NodeEmbedding(node_id=n.id, embedding=vec))
        backend.add_nodes(nodes)
        backend.store_embeddings(embeddings)

        results = backend.vector_search([1.0, 0.5, 0.3, 0.1, 0.0], limit=2)
        assert len(results) == 2

    def test_store_embeddings_upsert(self, backend: KuzuBackend) -> None:
        """Storing an embedding for the same node_id should update, not duplicate."""
        node = _make_node(name="upsert_func", file_path="src/u.py")
        backend.add_nodes([node])

        emb1 = NodeEmbedding(node_id=node.id, embedding=[1.0, 0.0])
        backend.store_embeddings([emb1])

        emb2 = NodeEmbedding(node_id=node.id, embedding=[0.0, 1.0])
        backend.store_embeddings([emb2])

        # Search with [0, 1] should find it with high similarity.
        results = backend.vector_search([0.0, 1.0], limit=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Fuzzy search tests
# ---------------------------------------------------------------------------


class TestFuzzySearch:
    """Levenshtein-based fuzzy name search."""

    def test_exact_name_returns_result(self, backend: KuzuBackend) -> None:
        """An exact name match (distance 0) should return a high score."""
        node = _make_node(name="validate_user", content="validates user")
        backend.add_nodes([node])

        results = backend.fuzzy_search("validate_user", limit=10)
        assert len(results) >= 1
        assert results[0].node_id == node.id
        assert results[0].score == 1.0  # distance 0 -> score 1.0

    def test_typo_within_distance(self, backend: KuzuBackend) -> None:
        """A misspelled query within max_distance should still find the node."""
        node = _make_node(name="validate_user", content="validates user")
        backend.add_nodes([node])

        # "validte_user" is 1 edit away from "validate_user"
        results = backend.fuzzy_search("validte_user", limit=10, max_distance=2)
        assert len(results) >= 1
        assert results[0].node_id == node.id
        assert results[0].score < 1.0  # distance > 0

    def test_typo_beyond_distance(self, backend: KuzuBackend) -> None:
        """A query that's too far away should not match."""
        node = _make_node(name="validate_user", content="validates user")
        backend.add_nodes([node])

        # "xyz_abc" is many edits away
        results = backend.fuzzy_search("xyz_abc", limit=10, max_distance=2)
        assert len(results) == 0

    def test_fuzzy_score_decreases_with_distance(self, backend: KuzuBackend) -> None:
        """Score should decrease as edit distance increases."""
        node = _make_node(name="process", content="")
        backend.add_nodes([node])

        exact = backend.fuzzy_search("process", limit=10)
        one_off = backend.fuzzy_search("procss", limit=10)  # 1 edit: missing 'e'
        assert len(exact) >= 1
        assert len(one_off) >= 1
        assert exact[0].score > one_off[0].score

    def test_fuzzy_limit(self, backend: KuzuBackend) -> None:
        """Only *limit* results should be returned."""
        nodes = [_make_node(name=f"func_{i}", file_path=f"src/f{i}.py") for i in range(5)]
        backend.add_nodes(nodes)

        results = backend.fuzzy_search("func_0", limit=2, max_distance=2)
        assert len(results) <= 2

    def test_fuzzy_result_fields(self, backend: KuzuBackend) -> None:
        """SearchResult should have populated fields."""
        node = _make_node(name="my_handler", file_path="src/handlers.py", content="handler body")
        backend.add_nodes([node])

        results = backend.fuzzy_search("my_handler", limit=10)
        assert len(results) >= 1
        r = results[0]
        assert r.node_name == "my_handler"
        assert r.file_path == "src/handlers.py"
        assert r.snippet != ""
