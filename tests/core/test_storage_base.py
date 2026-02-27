"""Tests for the storage backend abstraction layer."""

from __future__ import annotations

from axon.core.storage.base import NodeEmbedding, SearchResult, StorageBackend


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Verify SearchResult dataclass defaults and construction."""

    def test_creation_with_defaults(self) -> None:
        result = SearchResult(node_id="n1", score=0.95)
        assert result.node_id == "n1"
        assert result.score == 0.95
        assert result.node_name == ""
        assert result.file_path == ""
        assert result.label == ""
        assert result.snippet == ""

    def test_creation_with_all_fields(self) -> None:
        result = SearchResult(
            node_id="function:app.py:main",
            score=0.87,
            node_name="main",
            file_path="app.py",
            label="function",
            snippet="def main() -> None: ...",
        )
        assert result.node_id == "function:app.py:main"
        assert result.score == 0.87
        assert result.node_name == "main"
        assert result.file_path == "app.py"
        assert result.label == "function"
        assert result.snippet == "def main() -> None: ..."


# ---------------------------------------------------------------------------
# NodeEmbedding
# ---------------------------------------------------------------------------


class TestNodeEmbedding:
    """Verify NodeEmbedding dataclass defaults and construction."""

    def test_creation_with_defaults(self) -> None:
        emb = NodeEmbedding(node_id="n1")
        assert emb.node_id == "n1"
        assert emb.embedding == []

    def test_creation_with_data(self) -> None:
        vec = [0.1, 0.2, 0.3]
        emb = NodeEmbedding(node_id="n2", embedding=vec)
        assert emb.node_id == "n2"
        assert emb.embedding == [0.1, 0.2, 0.3]

    def test_embedding_default_is_independent(self) -> None:
        """Mutable default must not be shared across instances."""
        a = NodeEmbedding(node_id="a")
        b = NodeEmbedding(node_id="b")
        a.embedding.append(1.0)
        assert b.embedding == []


# ---------------------------------------------------------------------------
# StorageBackend protocol
# ---------------------------------------------------------------------------


class TestStorageBackend:
    """Verify the StorageBackend protocol is runtime-checkable."""

    def test_is_a_type(self) -> None:
        assert isinstance(StorageBackend, type)

    def test_runtime_checkable(self) -> None:
        """A class implementing all required methods should be recognised."""

        class _DummyBackend:
            def initialize(self, path):
                pass

            def close(self):
                pass

            def add_nodes(self, nodes):
                pass

            def add_relationships(self, rels):
                pass

            def remove_nodes_by_file(self, file_path):
                return 0

            def get_node(self, node_id):
                return None

            def get_callers(self, node_id):
                return []

            def get_callees(self, node_id):
                return []

            def get_type_refs(self, node_id):
                return []

            def get_callers_with_confidence(self, node_id):
                return []

            def get_callees_with_confidence(self, node_id):
                return []

            def traverse(self, start_id, depth):
                return []

            def traverse_with_depth(self, start_id, depth, direction="callers"):
                return []

            def get_process_memberships(self, node_ids):
                return {}

            def execute_raw(self, query):
                return None

            def exact_name_search(self, name, limit=5):
                return []

            def fts_search(self, query, limit):
                return []

            def fuzzy_search(self, query, limit, max_distance=2):
                return []

            def store_embeddings(self, embeddings):
                pass

            def vector_search(self, vector, limit):
                return []

            def get_indexed_files(self):
                return {}

            def bulk_load(self, graph):
                pass

            def load_graph(self):
                return None

            def delete_synthetic_nodes(self):
                pass

            def upsert_embeddings(self, embeddings):
                pass

            def update_dead_flags(self, dead_ids, alive_ids):
                pass

            def remove_relationships_by_type(self, rel_type):
                pass

            def rebuild_fts_indexes(self):
                pass

        assert isinstance(_DummyBackend(), StorageBackend)

    def test_non_conforming_class_fails(self) -> None:
        """A class missing required methods should NOT match the protocol."""

        class _Incomplete:
            pass

        assert not isinstance(_Incomplete(), StorageBackend)
