"""Tests for the in-memory KnowledgeGraph."""

from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a fresh, empty KnowledgeGraph."""
    return KnowledgeGraph()


def _make_node(
    label: NodeLabel = NodeLabel.FUNCTION,
    file_path: str = "src/app.py",
    name: str = "my_func",
) -> GraphNode:
    """Helper to build a GraphNode with a deterministic id."""
    return GraphNode(
        id=generate_id(label, file_path, name),
        label=label,
        name=name,
        file_path=file_path,
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


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------


class TestAddGetNode:
    def test_add_and_get_node(self, graph: KnowledgeGraph) -> None:
        node = _make_node()
        graph.add_node(node)
        assert graph.get_node(node.id) is node

    def test_get_node_returns_none_for_missing(self, graph: KnowledgeGraph) -> None:
        assert graph.get_node("nonexistent") is None

    def test_add_node_replaces_existing(self, graph: KnowledgeGraph) -> None:
        node_v1 = _make_node(name="foo")
        node_v2 = GraphNode(id=node_v1.id, label=NodeLabel.FUNCTION, name="foo_updated")
        graph.add_node(node_v1)
        graph.add_node(node_v2)
        assert graph.get_node(node_v1.id).name == "foo_updated"

    def test_nodes_property_returns_all(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        graph.add_node(n1)
        graph.add_node(n2)
        assert set(n.id for n in list(graph.iter_nodes())) == {n1.id, n2.id}


# ---------------------------------------------------------------------------
# Relationship CRUD
# ---------------------------------------------------------------------------


class TestAddRelationship:
    def test_add_relationship(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="caller")
        n2 = _make_node(name="callee")
        graph.add_node(n1)
        graph.add_node(n2)

        rel = _make_rel(n1.id, n2.id)
        graph.add_relationship(rel)

        assert list(graph.iter_relationships()) == [rel]

    def test_relationships_property_returns_all(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        n3 = _make_node(name="c")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        r1 = _make_rel(n1.id, n2.id, rel_id="r1")
        r2 = _make_rel(n2.id, n3.id, rel_id="r2")
        graph.add_relationship(r1)
        graph.add_relationship(r2)

        assert set(r.id for r in list(graph.iter_relationships())) == {"r1", "r2"}


# ---------------------------------------------------------------------------
# Remove node
# ---------------------------------------------------------------------------


class TestRemoveNode:
    def test_remove_existing_node_returns_true(self, graph: KnowledgeGraph) -> None:
        node = _make_node()
        graph.add_node(node)
        assert graph.remove_node(node.id) is True
        assert graph.get_node(node.id) is None

    def test_remove_missing_node_returns_false(self, graph: KnowledgeGraph) -> None:
        assert graph.remove_node("ghost") is False

    def test_remove_node_cascades_relationships(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        n3 = _make_node(name="c")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        # n1 -> n2, n2 -> n3, n1 -> n3
        r1 = _make_rel(n1.id, n2.id, rel_id="r1")
        r2 = _make_rel(n2.id, n3.id, rel_id="r2")
        r3 = _make_rel(n1.id, n3.id, rel_id="r3")
        graph.add_relationship(r1)
        graph.add_relationship(r2)
        graph.add_relationship(r3)

        # Removing n2 should cascade r1 (source=n2 in target) and r2 (source=n2)
        graph.remove_node(n2.id)

        remaining_ids = {r.id for r in list(graph.iter_relationships())}
        assert remaining_ids == {"r3"}, f"Expected only r3, got {remaining_ids}"


# ---------------------------------------------------------------------------
# Remove nodes by file
# ---------------------------------------------------------------------------


class TestRemoveNodesByFile:
    def test_removes_correct_count(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="func1", file_path="src/a.py")
        n2 = _make_node(name="func2", file_path="src/a.py")
        n3 = _make_node(name="func3", file_path="src/b.py")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        removed = graph.remove_nodes_by_file("src/a.py")
        assert removed == 2
        assert graph.get_node(n1.id) is None
        assert graph.get_node(n2.id) is None
        assert graph.get_node(n3.id) is not None

    def test_returns_zero_when_no_match(self, graph: KnowledgeGraph) -> None:
        assert graph.remove_nodes_by_file("nonexistent.py") == 0

    def test_cascades_relationships(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="f1", file_path="src/a.py")
        n2 = _make_node(name="f2", file_path="src/b.py")
        graph.add_node(n1)
        graph.add_node(n2)

        rel = _make_rel(n1.id, n2.id)
        graph.add_relationship(rel)

        graph.remove_nodes_by_file("src/a.py")
        assert list(graph.iter_relationships()) == []


# ---------------------------------------------------------------------------
# Query — by label / type
# ---------------------------------------------------------------------------


class TestQueryByLabelAndType:
    def test_get_nodes_by_label(self, graph: KnowledgeGraph) -> None:
        fn = _make_node(label=NodeLabel.FUNCTION, name="fn")
        cls = _make_node(label=NodeLabel.CLASS, name="Cls")
        graph.add_node(fn)
        graph.add_node(cls)

        result = graph.get_nodes_by_label(NodeLabel.FUNCTION)
        assert len(result) == 1
        assert result[0].id == fn.id

    def test_get_nodes_by_label_empty(self, graph: KnowledgeGraph) -> None:
        assert graph.get_nodes_by_label(NodeLabel.FILE) == []

    def test_get_relationships_by_type(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        graph.add_node(n1)
        graph.add_node(n2)

        calls_rel = _make_rel(n1.id, n2.id, RelType.CALLS, rel_id="c1")
        imports_rel = _make_rel(n1.id, n2.id, RelType.IMPORTS, rel_id="i1")
        graph.add_relationship(calls_rel)
        graph.add_relationship(imports_rel)

        result = graph.get_relationships_by_type(RelType.CALLS)
        assert len(result) == 1
        assert result[0].id == "c1"

    def test_get_relationships_by_type_empty(self, graph: KnowledgeGraph) -> None:
        assert graph.get_relationships_by_type(RelType.EXTENDS) == []


# ---------------------------------------------------------------------------
# Query — outgoing / incoming
# ---------------------------------------------------------------------------


class TestTraversal:
    def test_get_outgoing(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        n3 = _make_node(name="c")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        r1 = _make_rel(n1.id, n2.id, RelType.CALLS, rel_id="r1")
        r2 = _make_rel(n1.id, n3.id, RelType.IMPORTS, rel_id="r2")
        r3 = _make_rel(n2.id, n3.id, RelType.CALLS, rel_id="r3")
        graph.add_relationship(r1)
        graph.add_relationship(r2)
        graph.add_relationship(r3)

        # All outgoing from n1
        out = graph.get_outgoing(n1.id)
        assert set(r.id for r in out) == {"r1", "r2"}

        # Outgoing from n1 filtered to CALLS only
        out_calls = graph.get_outgoing(n1.id, rel_type=RelType.CALLS)
        assert len(out_calls) == 1
        assert out_calls[0].id == "r1"

    def test_get_incoming(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        n3 = _make_node(name="c")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        r1 = _make_rel(n1.id, n3.id, RelType.CALLS, rel_id="r1")
        r2 = _make_rel(n2.id, n3.id, RelType.IMPORTS, rel_id="r2")
        graph.add_relationship(r1)
        graph.add_relationship(r2)

        # All incoming to n3
        inc = graph.get_incoming(n3.id)
        assert set(r.id for r in inc) == {"r1", "r2"}

        # Incoming to n3 filtered to IMPORTS only
        inc_imports = graph.get_incoming(n3.id, rel_type=RelType.IMPORTS)
        assert len(inc_imports) == 1
        assert inc_imports[0].id == "r2"

    def test_get_outgoing_no_matches(self, graph: KnowledgeGraph) -> None:
        assert graph.get_outgoing("nonexistent") == []

    def test_get_incoming_no_matches(self, graph: KnowledgeGraph) -> None:
        assert graph.get_incoming("nonexistent") == []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_empty_graph(self, graph: KnowledgeGraph) -> None:
        assert graph.stats() == {"nodes": 0, "relationships": 0}

    def test_stats_with_data(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_relationship(_make_rel(n1.id, n2.id))

        assert graph.stats() == {"nodes": 2, "relationships": 1}

    def test_stats_after_removal(self, graph: KnowledgeGraph) -> None:
        n1 = _make_node(name="a")
        n2 = _make_node(name="b")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_relationship(_make_rel(n1.id, n2.id))

        graph.remove_node(n1.id)
        assert graph.stats() == {"nodes": 1, "relationships": 0}
