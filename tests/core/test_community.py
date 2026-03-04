"""Tests for the community detection phase (Phase 8)."""

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
from axon.core.ingestion.community import (
    export_to_igraph,
    generate_label,
    process_communities,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_function(
    graph: KnowledgeGraph,
    file_path: str,
    name: str,
    start_line: int = 1,
    end_line: int = 10,
) -> str:
    """Add a Function node and return its ID."""
    node_id = generate_id(NodeLabel.FUNCTION, file_path, name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=NodeLabel.FUNCTION,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
        )
    )
    return node_id


def _add_call(graph: KnowledgeGraph, source_id: str, target_id: str) -> None:
    """Add a CALLS relationship between two nodes."""
    rel_id = f"calls:{source_id}->{target_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source_id,
            target=target_id,
            properties={"confidence": 1.0},
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_cluster_graph() -> KnowledgeGraph:
    """Build a graph with two clear clusters connected by a single cross-edge.

    Cluster 1 (auth): validate, hash_password, check_token
        - validate -> hash_password
        - validate -> check_token
        - hash_password -> check_token

    Cluster 2 (data): query_db, format_result, cache_result
        - query_db -> format_result
        - query_db -> cache_result
        - format_result -> cache_result

    Cross-cluster: validate -> query_db
    """
    g = KnowledgeGraph()

    # Cluster 1: auth
    validate = _add_function(g, "src/auth/validate.py", "validate")
    hash_pw = _add_function(g, "src/auth/hash.py", "hash_password")
    check_tok = _add_function(g, "src/auth/token.py", "check_token")

    _add_call(g, validate, hash_pw)
    _add_call(g, validate, check_tok)
    _add_call(g, hash_pw, check_tok)

    # Cluster 2: data
    query_db = _add_function(g, "src/data/query.py", "query_db")
    format_res = _add_function(g, "src/data/format.py", "format_result")
    cache_res = _add_function(g, "src/data/cache.py", "cache_result")

    _add_call(g, query_db, format_res)
    _add_call(g, query_db, cache_res)
    _add_call(g, format_res, cache_res)

    # Cross-cluster edge
    _add_call(g, validate, query_db)

    return g


# ---------------------------------------------------------------------------
# test_export_to_igraph
# ---------------------------------------------------------------------------


class TestExportToIgraph:
    """export_to_igraph correctly converts the KnowledgeGraph call graph."""

    def test_export_to_igraph(self, two_cluster_graph: KnowledgeGraph) -> None:
        """Correct vertex and edge count for the two-cluster fixture."""
        ig_graph, index_map = export_to_igraph(two_cluster_graph)

        # 6 function nodes total.
        assert ig_graph.vcount() == 6
        # 7 CALLS edges (3 + 3 intra-cluster + 1 cross-cluster).
        assert ig_graph.ecount() == 7
        # Index map has one entry per vertex.
        assert len(index_map) == 6

    def test_export_to_igraph_empty(self) -> None:
        """Empty graph produces an empty igraph."""
        g = KnowledgeGraph()
        ig_graph, index_map = export_to_igraph(g)

        assert ig_graph.vcount() == 0
        assert ig_graph.ecount() == 0
        assert len(index_map) == 0


# ---------------------------------------------------------------------------
# test_process_communities
# ---------------------------------------------------------------------------


class TestProcessCommunities:
    """process_communities detects clusters and creates graph entities."""

    def test_process_communities_creates_nodes(self, two_cluster_graph: KnowledgeGraph) -> None:
        """Community nodes are created in the graph."""
        process_communities(two_cluster_graph)

        community_nodes = two_cluster_graph.get_nodes_by_label(NodeLabel.COMMUNITY)
        assert len(community_nodes) >= 1
        # Each community node must have the correct label.
        for node in community_nodes:
            assert node.label == NodeLabel.COMMUNITY
            assert node.name  # Non-empty label.
            assert "symbol_count" in node.properties
            assert "cohesion" in node.properties

    def test_process_communities_creates_member_of(self, two_cluster_graph: KnowledgeGraph) -> None:
        """MEMBER_OF relationships are created from members to communities."""
        process_communities(two_cluster_graph)

        member_rels = two_cluster_graph.get_relationships_by_type(RelType.MEMBER_OF)
        assert len(member_rels) >= 2  # At least some members assigned.

        # Every MEMBER_OF target must be a COMMUNITY node.
        for rel in member_rels:
            target_node = two_cluster_graph.get_node(rel.target)
            assert target_node is not None
            assert target_node.label == NodeLabel.COMMUNITY

        # Every MEMBER_OF source must be a callable node.
        callable_labels = {NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS}
        for rel in member_rels:
            source_node = two_cluster_graph.get_node(rel.source)
            assert source_node is not None
            assert source_node.label in callable_labels

    def test_process_communities_returns_count(self, two_cluster_graph: KnowledgeGraph) -> None:
        """Return value matches the number of Community nodes created."""
        count = process_communities(two_cluster_graph)

        community_nodes = two_cluster_graph.get_nodes_by_label(NodeLabel.COMMUNITY)
        assert count == len(community_nodes)
        assert count >= 1

    def test_process_communities_small_graph(self) -> None:
        """Graph with fewer than 3 callable nodes returns 0."""
        g = KnowledgeGraph()
        _add_function(g, "src/a.py", "foo")
        _add_function(g, "src/b.py", "bar")

        result = process_communities(g)

        assert result == 0
        assert len(g.get_nodes_by_label(NodeLabel.COMMUNITY)) == 0


# ---------------------------------------------------------------------------
# test_generate_label
# ---------------------------------------------------------------------------


class TestGenerateLabel:
    """generate_label produces human-readable community labels."""

    def test_generate_label_same_directory(self) -> None:
        """All members in one directory gives that directory name."""
        g = KnowledgeGraph()
        ids = [
            _add_function(g, "src/auth/validate.py", "validate"),
            _add_function(g, "src/auth/hash.py", "hash_password"),
            _add_function(g, "src/auth/token.py", "check_token"),
        ]

        label = generate_label(g, ids)
        assert label == "Auth"

    def test_generate_label_mixed_directories(self) -> None:
        """Members from different directories produce a combined label."""
        g = KnowledgeGraph()
        ids = [
            _add_function(g, "src/auth/validate.py", "validate"),
            _add_function(g, "src/auth/hash.py", "hash_password"),
            _add_function(g, "src/data/query.py", "query_db"),
        ]

        label = generate_label(g, ids)
        # Most common is "auth" (2 occurrences), second is "data".
        assert label == "Auth+data"

    def test_generate_label_no_file_paths(self) -> None:
        """Members with no file paths fall back to 'Cluster'."""
        g = KnowledgeGraph()
        node_id = "function:::orphan"
        g.add_node(
            GraphNode(
                id=node_id,
                label=NodeLabel.FUNCTION,
                name="orphan",
                file_path="",
            )
        )

        label = generate_label(g, [node_id])
        assert label == "Cluster"
