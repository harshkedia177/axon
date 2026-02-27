"""Tests for the KuzuDB storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.storage.base import NodeEmbedding
from axon.core.storage.kuzu_backend import KuzuBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> KuzuBackend:
    """Return a KuzuBackend initialised in a temporary directory."""
    db_path = tmp_path / "test_db"
    b = KuzuBackend()
    b.initialize(db_path)
    yield b
    b.close()


def _make_node(
    label: NodeLabel = NodeLabel.FUNCTION,
    file_path: str = "src/app.py",
    name: str = "my_func",
    content: str = "",
) -> GraphNode:
    """Helper to build a GraphNode with a deterministic id."""
    return GraphNode(
        id=generate_id(label, file_path, name),
        label=label,
        name=name,
        file_path=file_path,
        content=content,
    )


def _make_rel(
    source: str,
    target: str,
    rel_type: RelType = RelType.CALLS,
    rel_id: str | None = None,
) -> GraphRelationship:
    """Helper to build a GraphRelationship."""
    return GraphRelationship(
        id=rel_id or f"{rel_type.value}:{source}->{target}",
        type=rel_type,
        source=source,
        target=target,
    )


def _build_small_graph() -> KnowledgeGraph:
    """Build a small KnowledgeGraph with 2 functions and 1 CALLS relationship."""
    graph = KnowledgeGraph()

    caller = _make_node(name="caller", file_path="src/a.py")
    callee = _make_node(name="callee", file_path="src/a.py")
    graph.add_node(caller)
    graph.add_node(callee)

    rel = _make_rel(caller.id, callee.id)
    graph.add_relationship(rel)

    return graph


# ---------------------------------------------------------------------------
# Initialize and close
# ---------------------------------------------------------------------------


class TestInitializeAndClose:
    def test_initialize_creates_db(self, backend: KuzuBackend) -> None:
        """After initialize, internal handles should be set."""
        assert backend._db is not None
        assert backend._conn is not None

    def test_close_releases_handles(self, tmp_path: Path) -> None:
        b = KuzuBackend()
        b.initialize(tmp_path / "close_test")
        b.close()
        assert b._db is None
        assert b._conn is None


# ---------------------------------------------------------------------------
# bulk_load
# ---------------------------------------------------------------------------


class TestBulkLoad:
    def test_bulk_load_inserts_nodes_and_relationships(
        self, backend: KuzuBackend
    ) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        # Both function nodes should be retrievable.
        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        callee_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "callee")

        caller = backend.get_node(caller_id)
        callee = backend.get_node(callee_id)

        assert caller is not None
        assert caller.name == "caller"
        assert callee is not None
        assert callee.name == "callee"

    def test_bulk_load_replaces_existing(self, backend: KuzuBackend) -> None:
        """Calling bulk_load twice should not duplicate data."""
        graph = _build_small_graph()
        backend.bulk_load(graph)
        backend.bulk_load(graph)

        rows = backend.execute_raw("MATCH (n:Function) RETURN n.id")
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


class TestGetNode:
    def test_returns_correct_node(self, backend: KuzuBackend) -> None:
        node = _make_node(name="target_func", file_path="src/x.py")
        backend.add_nodes([node])

        result = backend.get_node(node.id)
        assert result is not None
        assert result.id == node.id
        assert result.name == "target_func"
        assert result.file_path == "src/x.py"
        assert result.label == NodeLabel.FUNCTION

    def test_returns_none_for_missing(self, backend: KuzuBackend) -> None:
        result = backend.get_node("function:nonexistent.py:ghost")
        assert result is None

    def test_returns_none_for_unknown_label(self, backend: KuzuBackend) -> None:
        result = backend.get_node("unknown_label:foo:bar")
        assert result is None

    def test_preserves_boolean_fields(self, backend: KuzuBackend) -> None:
        node = GraphNode(
            id=generate_id(NodeLabel.FUNCTION, "src/b.py", "entry"),
            label=NodeLabel.FUNCTION,
            name="entry",
            file_path="src/b.py",
            is_entry_point=True,
            is_exported=True,
        )
        backend.add_nodes([node])

        result = backend.get_node(node.id)
        assert result is not None
        assert result.is_entry_point is True
        assert result.is_exported is True
        assert result.is_dead is False


# ---------------------------------------------------------------------------
# get_callers / get_callees
# ---------------------------------------------------------------------------


class TestCallersAndCallees:
    def test_get_callers(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        callee_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "callee")
        callers = backend.get_callers(callee_id)

        assert len(callers) == 1
        assert callers[0].name == "caller"

    def test_get_callees(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        callees = backend.get_callees(caller_id)

        assert len(callees) == 1
        assert callees[0].name == "callee"

    def test_get_callers_empty(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        # The caller has no one calling it.
        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        callers = backend.get_callers(caller_id)
        assert callers == []

    def test_get_callees_empty(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        # The callee does not call anyone.
        callee_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "callee")
        callees = backend.get_callees(callee_id)
        assert callees == []


# ---------------------------------------------------------------------------
# execute_raw
# ---------------------------------------------------------------------------


class TestExecuteRaw:
    def test_simple_cypher(self, backend: KuzuBackend) -> None:
        backend.add_nodes([_make_node(name="raw_test")])

        rows = backend.execute_raw("MATCH (n:Function) RETURN n.name")
        assert len(rows) == 1
        assert rows[0][0] == "raw_test"

    def test_return_expression(self, backend: KuzuBackend) -> None:
        rows = backend.execute_raw("RETURN 1 + 2 AS result")
        assert rows == [[3]]


# ---------------------------------------------------------------------------
# get_indexed_files
# ---------------------------------------------------------------------------


class TestGetIndexedFiles:
    def test_returns_empty_initially(self, backend: KuzuBackend) -> None:
        result = backend.get_indexed_files()
        assert result == {}

    def test_returns_files_after_insert(self, backend: KuzuBackend) -> None:
        file_node = _make_node(
            label=NodeLabel.FILE,
            file_path="src/main.py",
            name="main.py",
            content="print('hello')",
        )
        backend.add_nodes([file_node])

        result = backend.get_indexed_files()
        assert "src/main.py" in result
        # The hash should be the sha256 of the content.
        import hashlib

        expected_hash = hashlib.sha256(b"print('hello')").hexdigest()
        assert result["src/main.py"] == expected_hash


# ---------------------------------------------------------------------------
# remove_nodes_by_file
# ---------------------------------------------------------------------------


class TestRemoveNodesByFile:
    def test_removes_matching_nodes(self, backend: KuzuBackend) -> None:
        n1 = _make_node(name="f1", file_path="src/a.py")
        n2 = _make_node(name="f2", file_path="src/a.py")
        n3 = _make_node(name="f3", file_path="src/b.py")
        backend.add_nodes([n1, n2, n3])

        backend.remove_nodes_by_file("src/a.py")

        assert backend.get_node(n1.id) is None
        assert backend.get_node(n2.id) is None
        assert backend.get_node(n3.id) is not None

    def test_returns_zero_for_no_match(self, backend: KuzuBackend) -> None:
        result = backend.remove_nodes_by_file("nonexistent.py")
        assert result == 0


# ---------------------------------------------------------------------------
# traverse
# ---------------------------------------------------------------------------


class TestTraverse:
    def test_traverse_one_hop(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        nodes = backend.traverse(caller_id, depth=1, direction="callees")

        assert len(nodes) == 1
        assert nodes[0].name == "callee"

    def test_traverse_zero_depth(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        nodes = backend.traverse(caller_id, depth=0, direction="callees")
        assert nodes == []


# ---------------------------------------------------------------------------
# add_nodes with different labels
# ---------------------------------------------------------------------------


class TestMultipleLabels:
    def test_class_and_function(self, backend: KuzuBackend) -> None:
        fn = _make_node(label=NodeLabel.FUNCTION, name="my_fn", file_path="src/c.py")
        cls = _make_node(label=NodeLabel.CLASS, name="MyClass", file_path="src/c.py")
        backend.add_nodes([fn, cls])

        assert backend.get_node(fn.id) is not None
        assert backend.get_node(cls.id) is not None
        assert backend.get_node(fn.id).label == NodeLabel.FUNCTION
        assert backend.get_node(cls.id).label == NodeLabel.CLASS


# ---------------------------------------------------------------------------
# load_graph
# ---------------------------------------------------------------------------


class TestLoadGraph:
    def test_round_trips_nodes_and_relationships(self, backend: KuzuBackend) -> None:
        """Store 3 nodes and 2 relationships, load_graph, verify counts and existence."""
        n1 = _make_node(name="alpha", file_path="src/a.py")
        n2 = _make_node(name="beta", file_path="src/a.py")
        n3 = _make_node(label=NodeLabel.CLASS, name="Gamma", file_path="src/a.py")
        backend.add_nodes([n1, n2, n3])

        r1 = _make_rel(n1.id, n2.id, RelType.CALLS)
        r2 = _make_rel(n1.id, n3.id, RelType.CALLS)
        backend.add_relationships([r1, r2])

        graph = backend.load_graph()

        assert graph.node_count == 3
        assert graph.relationship_count == 2
        assert graph.get_node(n1.id) is not None
        assert graph.get_node(n2.id) is not None
        assert graph.get_node(n3.id) is not None

    def test_preserves_node_properties(self, backend: KuzuBackend) -> None:
        """Store a node with boolean flags and signature, verify they survive round-trip."""
        node = GraphNode(
            id=generate_id(NodeLabel.FUNCTION, "src/d.py", "special"),
            label=NodeLabel.FUNCTION,
            name="special",
            file_path="src/d.py",
            signature="def special() -> bool",
            is_dead=True,
            is_entry_point=True,
        )
        backend.add_nodes([node])

        graph = backend.load_graph()
        loaded = graph.get_node(node.id)

        assert loaded is not None
        assert loaded.name == "special"
        assert loaded.signature == "def special() -> bool"
        assert loaded.is_dead is True
        assert loaded.is_entry_point is True
        assert loaded.is_exported is False

    def test_empty_storage_returns_empty_graph(self, backend: KuzuBackend) -> None:
        """Loading from an empty database returns an empty KnowledgeGraph."""
        graph = backend.load_graph()

        assert graph.node_count == 0
        assert graph.relationship_count == 0


# ---------------------------------------------------------------------------
# delete_synthetic_nodes
# ---------------------------------------------------------------------------


class TestDeleteSyntheticNodes:
    def test_removes_community_and_process_keeps_function(
        self, backend: KuzuBackend
    ) -> None:
        """Store fn + community + process with edges. After delete, only fn remains."""
        fn = _make_node(name="real_func", file_path="src/a.py")
        comm = _make_node(
            label=NodeLabel.COMMUNITY, name="comm_1", file_path=""
        )
        proc = _make_node(
            label=NodeLabel.PROCESS, name="proc_1", file_path=""
        )
        backend.add_nodes([fn, comm, proc])

        # Add MEMBER_OF edge (fn -> community) and STEP_IN_PROCESS (fn -> process).
        r1 = _make_rel(fn.id, comm.id, RelType.MEMBER_OF)
        r2 = _make_rel(fn.id, proc.id, RelType.STEP_IN_PROCESS)
        backend.add_relationships([r1, r2])

        backend.delete_synthetic_nodes()

        graph = backend.load_graph()
        # Only the function node should survive.
        assert graph.node_count == 1
        assert graph.get_node(fn.id) is not None
        assert graph.get_node(comm.id) is None
        assert graph.get_node(proc.id) is None
        # All relationships should be gone (targets deleted).
        assert graph.relationship_count == 0


# ---------------------------------------------------------------------------
# upsert_embeddings
# ---------------------------------------------------------------------------


class TestUpsertEmbeddings:
    def test_upserts_without_wiping(self, backend: KuzuBackend) -> None:
        """store_embeddings + upsert_embeddings should result in both existing."""
        emb_a = NodeEmbedding(node_id="function:src/a.py:alpha", embedding=[1.0, 2.0])
        emb_b = NodeEmbedding(node_id="function:src/a.py:beta", embedding=[3.0, 4.0])

        backend.store_embeddings([emb_a])
        backend.upsert_embeddings([emb_b])

        rows = backend.execute_raw(
            "MATCH (e:Embedding) RETURN e.node_id ORDER BY e.node_id"
        )
        node_ids = [r[0] for r in rows]
        assert "function:src/a.py:alpha" in node_ids
        assert "function:src/a.py:beta" in node_ids

    def test_updates_existing_embedding(self, backend: KuzuBackend) -> None:
        """Upserting an existing node_id should update the vector, not duplicate."""
        emb = NodeEmbedding(node_id="function:src/a.py:alpha", embedding=[1.0, 2.0])
        backend.store_embeddings([emb])

        updated = NodeEmbedding(
            node_id="function:src/a.py:alpha", embedding=[9.0, 8.0]
        )
        backend.upsert_embeddings([updated])

        rows = backend.execute_raw(
            "MATCH (e:Embedding) WHERE e.node_id = 'function:src/a.py:alpha' "
            "RETURN e.vec"
        )
        assert len(rows) == 1
        assert rows[0][0][0] == 9.0
        assert rows[0][0][1] == 8.0


# ---------------------------------------------------------------------------
# update_dead_flags
# ---------------------------------------------------------------------------


class TestUpdateDeadFlags:
    def test_sets_dead_and_alive(self, backend: KuzuBackend) -> None:
        """Mark one node dead and another alive, verify via get_node."""
        n1 = _make_node(name="func_a", file_path="src/a.py")
        n2 = _make_node(name="func_b", file_path="src/a.py")
        backend.add_nodes([n1, n2])

        backend.update_dead_flags(dead_ids={n1.id}, alive_ids={n2.id})

        dead_node = backend.get_node(n1.id)
        alive_node = backend.get_node(n2.id)
        assert dead_node is not None
        assert dead_node.is_dead is True
        assert alive_node is not None
        assert alive_node.is_dead is False


# ---------------------------------------------------------------------------
# remove_relationships_by_type
# ---------------------------------------------------------------------------


class TestRemoveRelationshipsByType:
    def test_removes_only_specified_type(self, backend: KuzuBackend) -> None:
        """Store CALLS and COUPLED_WITH. Remove COUPLED_WITH. Verify CALLS survives."""
        n1 = _make_node(name="func_x", file_path="src/a.py")
        n2 = _make_node(name="func_y", file_path="src/a.py")
        backend.add_nodes([n1, n2])

        calls_rel = _make_rel(n1.id, n2.id, RelType.CALLS)
        coupled_rel = _make_rel(n1.id, n2.id, RelType.COUPLED_WITH)
        backend.add_relationships([calls_rel, coupled_rel])

        backend.remove_relationships_by_type(RelType.COUPLED_WITH)

        graph = backend.load_graph()
        rel_types = [r.type for r in graph.iter_relationships()]
        assert RelType.CALLS in rel_types
        assert RelType.COUPLED_WITH not in rel_types
