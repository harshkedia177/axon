"""Storage backend abstraction for Axon.

Defines the :class:`StorageBackend` protocol that all concrete storage
implementations (KuzuDB, Neo4j, in-memory, etc.) must satisfy, along with
supporting data classes for search results and embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, RelType


@dataclass
class SearchResult:
    """A single result from a full-text or vector search."""

    node_id: str
    score: float
    node_name: str = ""
    file_path: str = ""
    label: str = ""
    snippet: str = ""

EMBEDDING_DIMENSIONS: int = 384
"""Number of dimensions expected for all embedding vectors."""



@dataclass
class NodeEmbedding:
    """An embedding vector associated with a graph node."""

    node_id: str
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.embedding and len(self.embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected embedding of {EMBEDDING_DIMENSIONS} dimensions, "
                f"got {len(self.embedding)}"
            )

@runtime_checkable
class StorageBackend(Protocol):
    """Protocol that every Axon storage backend must implement.

    Covers the full lifecycle of graph persistence: initialisation,
    CRUD operations on nodes and relationships, querying, full-text
    search, vector search, and incremental re-indexing support.
    """

    def initialize(self, path: Path) -> None:
        """Open or create the backing store at *path*."""
        ...

    def close(self) -> None:
        """Release resources held by the backend."""
        ...

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Insert or upsert a batch of nodes."""
        ...

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        """Insert or upsert a batch of relationships."""
        ...

    def remove_nodes_by_file(self, file_path: str) -> int:
        """Remove all nodes originating from *file_path*.

        Returns:
            The number of nodes removed.
        """
        ...

    def get_inbound_cross_file_edges(
        self, file_path: str, exclude_source_files: set[str] | None = None,
    ) -> list[GraphRelationship]:
        """Return inbound edges where the target is in *file_path* and the source is not.

        Args:
            file_path: Target file whose inbound edges to collect.
            exclude_source_files: Source file paths to skip.
        """
        ...

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a single node by ID, or ``None`` if not found."""
        ...

    def get_callers(self, node_id: str) -> list[GraphNode]:
        """Return nodes that call the node identified by *node_id*."""
        ...

    def get_callees(self, node_id: str) -> list[GraphNode]:
        """Return nodes called by the node identified by *node_id*."""
        ...

    def get_type_refs(self, node_id: str) -> list[GraphNode]:
        """Return nodes that reference the type identified by *node_id*."""
        ...

    def get_callers_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return ``(node, confidence)`` pairs for all nodes that CALL *node_id*."""
        ...

    def get_callees_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return ``(node, confidence)`` pairs for all nodes called by *node_id*."""
        ...

    def traverse(self, start_id: str, depth: int, direction: str = "callers") -> list[GraphNode]:
        """Breadth-first traversal up to *depth* hops from *start_id*.

        Args:
            direction: ``"callers"`` follows incoming CALLS (blast radius),
                       ``"callees"`` follows outgoing CALLS (dependencies).
        """
        ...

    def traverse_with_depth(
        self, start_id: str, depth: int, direction: str = "callers"
    ) -> list[tuple[GraphNode, int]]:
        """BFS traversal returning ``(node, hop_depth)`` pairs.

        Same semantics as :meth:`traverse` but preserves the hop distance
        (1-based) so callers can group results by proximity.
        """
        ...

    def get_process_memberships(self, node_ids: list[str]) -> dict[str, str]:
        """Return ``{node_id: process_name}`` for nodes belonging to a Process."""
        ...

    def execute_raw(self, query: str) -> Any:
        """Execute a raw backend-specific query string."""
        ...

    def exact_name_search(self, name: str, limit: int = 5) -> list[SearchResult]:
        """Search for nodes with an exact name match."""
        ...

    def fts_search(self, query: str, limit: int) -> list[SearchResult]:
        """Full-text search across indexed node content."""
        ...

    def fuzzy_search(
        self, query: str, limit: int, max_distance: int = 2
    ) -> list[SearchResult]:
        """Fuzzy name search by edit distance."""
        ...

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Persist embedding vectors for the given nodes."""
        ...

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """Find the closest nodes to *vector* by cosine similarity."""
        ...

    def get_indexed_files(self) -> dict[str, str]:
        """Return a mapping of ``{file_path: content_hash}`` for all indexed files."""
        ...

    def load_graph(self) -> KnowledgeGraph:
        """Reconstruct a full :class:`KnowledgeGraph` from the backing store."""
        ...

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """Replace the entire store contents with *graph*."""
        ...

    def delete_synthetic_nodes(self) -> None:
        """Remove all COMMUNITY and PROCESS nodes and their relationships."""
        ...

    def upsert_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Insert or update embeddings without wiping existing ones."""
        ...

    def update_dead_flags(self, dead_ids: set[str], alive_ids: set[str]) -> None:
        """Set is_dead=True on *dead_ids* and is_dead=False on *alive_ids*."""
        ...

    def remove_relationships_by_type(self, rel_type: RelType) -> None:
        """Delete all relationships of a specific type."""
        ...

    def get_file_index(self) -> dict[str, str]:
        """Return ``{file_path: node_id}`` for all File nodes."""
        ...

    def get_symbol_name_index(self) -> dict[str, list[str]]:
        """Return ``{symbol_name: [node_id, ...]}`` for callable/type symbols."""
        ...

    def rebuild_fts_indexes(self) -> None:
        """Drop and recreate all FTS indexes after bulk data changes."""
        ...
