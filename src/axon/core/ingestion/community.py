"""Phase 8: Community detection for Axon.

Uses igraph + leidenalg to partition the call graph into functional clusters
(communities). Each community groups tightly-connected symbols that likely
belong to the same logical feature or module.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import PurePosixPath

import igraph as ig
import leidenalg

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)

logger = logging.getLogger(__name__)

_CALLABLE_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)


def export_to_igraph(
    graph: KnowledgeGraph,
) -> tuple[ig.Graph, dict[int, str]]:
    """Extract the call graph from *graph* and build an igraph representation.

    Only Function, Method, and Class nodes are included. Only CALLS
    relationships between those nodes are used as edges.

    Args:
        graph: The Axon knowledge graph.

    Returns:
        A tuple of ``(igraph_graph, vertex_index_to_node_id)`` where the
        mapping connects igraph vertex indices back to Axon node IDs.
    """
    node_id_to_index: dict[str, int] = {}
    index_to_node_id: dict[int, str] = {}

    for label in _CALLABLE_LABELS:
        for node in graph.get_nodes_by_label(label):
            idx = len(node_id_to_index)
            node_id_to_index[node.id] = idx
            index_to_node_id[idx] = node.id

    num_vertices = len(node_id_to_index)

    edge_list: list[tuple[int, int]] = []
    for rel in graph.get_relationships_by_type(RelType.CALLS):
        src_idx = node_id_to_index.get(rel.source)
        tgt_idx = node_id_to_index.get(rel.target)
        if src_idx is not None and tgt_idx is not None:
            edge_list.append((src_idx, tgt_idx))

    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(num_vertices)
    ig_graph.add_edges(edge_list)

    return ig_graph, index_to_node_id


def generate_label(graph: KnowledgeGraph, member_ids: list[str]) -> str:
    """Generate a heuristic label for a community based on member file paths.

    Strategy:
    - Extract the parent directory from each member's ``file_path``.
    - If all members share the same directory, use that directory name.
    - Otherwise, combine the two most frequent directories with ``+``.
    - Capitalize and clean up the result.

    Falls back to ``"cluster"`` if no file paths are available.

    Args:
        graph: The knowledge graph (used to look up member nodes).
        member_ids: List of node IDs belonging to this community.

    Returns:
        A human-readable label string.
    """
    directories: list[str] = []
    for nid in member_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path:
            parent = PurePosixPath(node.file_path).parent.name
            if parent:
                directories.append(parent)

    if not directories:
        return "Cluster"

    counts = Counter(directories)
    most_common = counts.most_common(2)

    if len(most_common) == 1 or most_common[0][0] == most_common[-1][0]:
        # All members in the same directory.
        return most_common[0][0].capitalize()

    # Mixed directories: combine top two.
    label = f"{most_common[0][0]}+{most_common[1][0]}"
    return label.capitalize()


def process_communities(
    graph: KnowledgeGraph,
    min_community_size: int = 2,
) -> int:
    """Detect communities in the call graph and add them to the knowledge graph.

    Uses the Leiden algorithm with modularity-based vertex partitioning.

    For each detected community that meets the minimum size threshold:
    - A :attr:`NodeLabel.COMMUNITY` node is created with a generated label
      and metadata (cohesion score, symbol count).
    - :attr:`RelType.MEMBER_OF` relationships are created from each member
      symbol to the community node.

    Args:
        graph: The knowledge graph to analyze and augment.
        min_community_size: Minimum number of members for a community to be
            created. Communities smaller than this are skipped.

    Returns:
        The number of community nodes created.
    """
    ig_graph, index_to_node_id = export_to_igraph(graph)

    if ig_graph.vcount() < 3:
        logger.debug(
            "Call graph too small for community detection (%d nodes), skipping.",
            ig_graph.vcount(),
        )
        return 0

    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    modularity_score = partition.modularity

    community_count = 0
    for i, members in enumerate(partition):
        if len(members) < min_community_size:
            continue

        member_ids = [index_to_node_id[idx] for idx in members]

        community_id = generate_id(NodeLabel.COMMUNITY, f"community_{i}")
        label = generate_label(graph, member_ids)

        community_node = GraphNode(
            id=community_id,
            label=NodeLabel.COMMUNITY,
            name=label,
            properties={
                "cohesion": modularity_score,
                "symbol_count": len(member_ids),
            },
        )
        graph.add_node(community_node)

        for member_id in member_ids:
            rel_id = f"member_of:{member_id}->{community_id}"
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.MEMBER_OF,
                    source=member_id,
                    target=community_id,
                )
            )

        community_count += 1
        logger.info(
            "Community %d: %r with %d members (modularity=%.3f)",
            i,
            label,
            len(member_ids),
            modularity_score,
        )

    logger.info("Community detection complete: %d communities created.", community_count)
    return community_count
