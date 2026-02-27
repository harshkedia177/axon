"""KuzuDB storage backend for Axon.

Implements the :class:`StorageBackend` protocol using KuzuDB, an embedded
graph database that speaks Cypher. Each :class:`NodeLabel` maps to a
separate node table, and a single ``CodeRelation`` relationship table group
covers all source-to-target combinations.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import kuzu

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from axon.core.storage.base import NodeEmbedding, SearchResult

logger = logging.getLogger(__name__)

_NODE_TABLE_NAMES: list[str] = [label.name.title().replace("_", "") for label in NodeLabel]

_LABEL_TO_TABLE: dict[str, str] = {
    label.value: label.name.title().replace("_", "") for label in NodeLabel
}

_LABEL_MAP: dict[str, NodeLabel] = {label.value: label for label in NodeLabel}

_REL_TYPE_MAP: dict[str, RelType] = {rt.value: rt for rt in RelType}

_SEARCHABLE_TABLES: list[str] = [
    t for t in _NODE_TABLE_NAMES
    if t not in ("Folder", "Community", "Process")
]

_NODE_PROPERTIES = (
    "id STRING, "
    "name STRING, "
    "file_path STRING, "
    "start_line INT64, "
    "end_line INT64, "
    "content STRING, "
    "signature STRING, "
    "language STRING, "
    "class_name STRING, "
    "is_dead BOOL, "
    "is_entry_point BOOL, "
    "is_exported BOOL, "
    "PRIMARY KEY (id)"
)

_REL_PROPERTIES = (
    "rel_type STRING, "
    "confidence DOUBLE, "
    "role STRING, "
    "step_number INT64, "
    "strength DOUBLE, "
    "co_changes INT64, "
    "symbols STRING"
)

def _escape(value: str) -> str:
    """Escape a string for safe inclusion in a Cypher literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")

def _table_for_id(node_id: str) -> str | None:
    """Extract the table name from a node ID by mapping its label prefix."""
    prefix = node_id.split(":", 1)[0]
    return _LABEL_TO_TABLE.get(prefix)

_EMBEDDING_PROPERTIES = "node_id STRING, vec DOUBLE[], PRIMARY KEY(node_id)"

class KuzuBackend:
    """StorageBackend implementation backed by KuzuDB.

    Usage::

        backend = KuzuBackend()
        backend.initialize(Path("/tmp/axon_db"))
        backend.bulk_load(graph)
        node = backend.get_node("function:src/app.py:main")
        backend.close()
    """

    def __init__(self) -> None:
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    def initialize(self, path: Path, *, read_only: bool = False) -> None:
        """Open or create the KuzuDB database at *path* and set up the schema.

        Args:
            path: Filesystem path to the KuzuDB database directory.
            read_only: If ``True``, open the database in read-only mode.
                This allows multiple concurrent readers (e.g. MCP server
                instances) without lock conflicts.  Schema creation is
                skipped since the database must already exist.
        """
        self._db = kuzu.Database(str(path), read_only=read_only)
        self._conn = kuzu.Connection(self._db)
        if not read_only:
            self._create_schema()

    def close(self) -> None:
        """Release the connection and database handles.

        Explicitly deletes the connection and database objects to ensure
        KuzuDB releases file locks and flushes data.
        """
        if self._conn is not None:
            try:
                del self._conn
            except Exception:
                pass
            self._conn = None
        if self._db is not None:
            try:
                del self._db
            except Exception:
                pass
            self._db = None

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Insert nodes into their respective label tables."""
        for node in nodes:
            self._insert_node(node)

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        """Insert relationships by matching source and target nodes."""
        for rel in rels:
            self._insert_relationship(rel)

    def remove_nodes_by_file(self, file_path: str) -> int:
        """Delete all nodes whose ``file_path`` matches across every table.

        Returns:
            Always 0 — exact count is not tracked for performance.
        """
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            try:
                self._conn.execute(
                    f"MATCH (n:{table}) WHERE n.file_path = $fp DETACH DELETE n",
                    parameters={"fp": file_path},
                )
            except Exception:
                logger.debug("Failed to remove nodes from table %s", table, exc_info=True)
        return 0

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a single node by ID, or ``None`` if not found."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return None

        query = f"MATCH (n:{table}) WHERE n.id = $nid RETURN n.*"
        try:
            result = self._conn.execute(query, parameters={"nid": node_id})
            if result.has_next():
                row = result.get_next()
                return self._row_to_node(row, node_id)
        except Exception:
            logger.debug("get_node failed for %s", node_id, exc_info=True)
        return None

    def get_callers(self, node_id: str) -> list[GraphNode]:
        """Return nodes that CALL the node identified by *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller)-[r:CodeRelation]->(callee:{table}) "
            f"WHERE callee.id = $nid AND r.rel_type = 'calls' "
            f"RETURN caller.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_callees(self, node_id: str) -> list[GraphNode]:
        """Return nodes called by the node identified by *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller:{table})-[r:CodeRelation]->(callee) "
            f"WHERE caller.id = $nid AND r.rel_type = 'calls' "
            f"RETURN callee.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_type_refs(self, node_id: str) -> list[GraphNode]:
        """Return nodes referenced via USES_TYPE from *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (src:{table})-[r:CodeRelation]->(tgt) "
            f"WHERE src.id = $nid AND r.rel_type = 'uses_type' "
            f"RETURN tgt.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_callers_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return ``(node, confidence)`` for all callers of *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []
        query = (
            f"MATCH (caller)-[r:CodeRelation]->(callee:{table}) "
            f"WHERE callee.id = $nid AND r.rel_type = 'calls' "
            f"RETURN caller.*, r.confidence"
        )
        return self._query_nodes_with_confidence(query, parameters={"nid": node_id})

    def get_callees_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return ``(node, confidence)`` for all callees of *node_id*."""
        assert self._conn is not None
        table = _table_for_id(node_id)
        if table is None:
            return []
        query = (
            f"MATCH (caller:{table})-[r:CodeRelation]->(callee) "
            f"WHERE caller.id = $nid AND r.rel_type = 'calls' "
            f"RETURN callee.*, r.confidence"
        )
        return self._query_nodes_with_confidence(query, parameters={"nid": node_id})

    _MAX_BFS_DEPTH = 10

    def traverse(self, start_id: str, depth: int, direction: str = "callers") -> list[GraphNode]:
        """BFS traversal through CALLS edges — flat result list (no depth info)."""
        return [node for node, _ in self.traverse_with_depth(start_id, depth, direction)]

    def traverse_with_depth(
        self, start_id: str, depth: int, direction: str = "callers"
    ) -> list[tuple[GraphNode, int]]:
        """BFS traversal returning ``(node, hop_depth)`` pairs.

        ``hop_depth`` is 1-based: direct callers/callees are depth 1.

        Args:
            direction: ``"callers"`` follows incoming CALLS (blast radius),
                       ``"callees"`` follows outgoing CALLS (dependencies).
        """
        assert self._conn is not None
        depth = min(depth, self._MAX_BFS_DEPTH)
        if _table_for_id(start_id) is None:
            return []

        visited: set[str] = set()
        result_list: list[tuple[GraphNode, int]] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != start_id:
                node = self.get_node(current_id)
                if node is not None:
                    result_list.append((node, current_depth))

            if current_depth < depth:
                neighbors = (
                    self.get_callers(current_id)
                    if direction == "callers"
                    else self.get_callees(current_id)
                )
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, current_depth + 1))

        return result_list

    def get_process_memberships(self, node_ids: list[str]) -> dict[str, str]:
        """Return ``{node_id: process_name}`` for nodes in any Process.

        Uses parameterized IN clause to safely query all node IDs at once.
        """
        assert self._conn is not None
        if not node_ids:
            return {}

        mapping: dict[str, str] = {}
        try:
            result = self._conn.execute(
                "MATCH (n)-[r:CodeRelation]->(p:Process) "
                "WHERE n.id IN $ids AND r.rel_type = 'step_in_process' "
                "RETURN n.id, p.name",
                parameters={"ids": node_ids},
            )
            while result.has_next():
                row = result.get_next()
                nid = row[0] if row else ""
                pname = row[1] if len(row) > 1 else ""
                if nid and pname and nid not in mapping:
                    mapping[nid] = pname
        except Exception:
            logger.debug("get_process_memberships failed", exc_info=True)
        return mapping

    def execute_raw(self, query: str) -> list[list[Any]]:
        """Execute a raw Cypher query and return all result rows."""
        assert self._conn is not None
        result = self._conn.execute(query)
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def exact_name_search(self, name: str, limit: int = 5) -> list[SearchResult]:
        """Search for nodes with an exact name match across all searchable tables.

        Returns results sorted by label priority (functions/methods first),
        preferring source files over test files.
        """
        assert self._conn is not None
        candidates: list[SearchResult] = []

        for table in _SEARCHABLE_TABLES:
            cypher = (
                f"MATCH (n:{table}) WHERE n.name = $name "
                f"RETURN n.id, n.name, n.file_path, n.content, n.signature "
                f"LIMIT {limit}"
            )
            try:
                result = self._conn.execute(cypher, parameters={"name": name})
                while result.has_next():
                    row = result.get_next()
                    node_id = row[0] or ""
                    node_name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    signature = row[4] or ""
                    label_prefix = node_id.split(":", 1)[0] if node_id else ""
                    snippet = content[:200] if content else signature[:200]
                    score = 2.0 if "/tests/" not in file_path else 1.0
                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=score,
                            node_name=node_name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=snippet,
                        )
                    )
            except Exception:
                logger.debug("exact_name_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def fts_search(self, query: str, limit: int) -> list[SearchResult]:
        """BM25 full-text search using KuzuDB's native FTS extension.

        Searches across all node tables using pre-built FTS indexes on
        ``name``, ``content``, and ``signature`` fields.  Results are
        ranked by BM25 relevance score.

        Returns the top *limit* results sorted by score descending.
        """
        assert self._conn is not None
        escaped_q = _escape(query)
        candidates: list[SearchResult] = []

        for table in _SEARCHABLE_TABLES:
            idx_name = f"{table.lower()}_fts"
            cypher = (
                f"CALL QUERY_FTS_INDEX('{table}', '{idx_name}', '{escaped_q}') "
                f"RETURN node.id, node.name, node.file_path, node.content, "
                f"node.signature, score "
                f"ORDER BY score DESC LIMIT {limit}"
            )
            try:
                result = self._conn.execute(cypher)
                while result.has_next():
                    row = result.get_next()
                    node_id = row[0] or ""
                    name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    signature = row[4] or ""
                    bm25_score = float(row[5]) if row[5] is not None else 0.0

                    # Demote test file results — mirrors exact_name_search penalty.
                    if "/tests/" in file_path or "/test_" in file_path:
                        bm25_score *= 0.5

                    label_prefix = node_id.split(":", 1)[0] if node_id else ""

                    # Boost top-level definitions in source files.
                    if label_prefix in ("function", "class") and "/tests/" not in file_path:
                        bm25_score *= 1.2

                    snippet = content[:200] if content else signature[:200]

                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=bm25_score,
                            node_name=name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=snippet,
                        )
                    )
            except Exception:
                logger.debug("fts_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def fuzzy_search(
        self, query: str, limit: int, max_distance: int = 2
    ) -> list[SearchResult]:
        """Fuzzy name search using Levenshtein edit distance.

        Scans all node tables for symbols whose name is within
        *max_distance* edits of *query*.  Converts edit distance to a
        score (0 edits = 1.0, *max_distance* edits = 0.3).
        """
        assert self._conn is not None
        escaped_q = _escape(query.lower())
        candidates: list[SearchResult] = []

        for table in _SEARCHABLE_TABLES:
            cypher = (
                f"MATCH (n:{table}) "
                f"WHERE levenshtein(lower(n.name), '{escaped_q}') <= {max_distance} "
                f"RETURN n.id, n.name, n.file_path, n.content, "
                f"levenshtein(lower(n.name), '{escaped_q}') AS dist "
                f"ORDER BY dist LIMIT {limit}"
            )
            try:
                result = self._conn.execute(cypher)
                while result.has_next():
                    row = result.get_next()
                    node_id = row[0] or ""
                    name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    dist = int(row[4]) if row[4] is not None else max_distance

                    score = max(0.3, 1.0 - (dist * 0.3))
                    label_prefix = node_id.split(":", 1)[0] if node_id else ""

                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=score,
                            node_name=name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=content[:200] if content else "",
                        )
                    )
            except Exception:
                logger.debug("fuzzy_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Persist embedding vectors into the Embedding node table.

        Attempts batch CSV COPY FROM first, falls back to individual MERGE.
        """
        assert self._conn is not None
        if not embeddings:
            return

        if self._bulk_store_embeddings_csv(embeddings):
            return

        for emb in embeddings:
            try:
                self._conn.execute(
                    "MERGE (e:Embedding {node_id: $nid}) SET e.vec = $vec",
                    parameters={"nid": emb.node_id, "vec": emb.embedding},
                )
            except Exception:
                logger.debug(
                    "store_embeddings failed for node %s", emb.node_id, exc_info=True
                )

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """Find the closest nodes to *vector* using native ``array_cosine_similarity``.

        Computes cosine similarity directly in KuzuDB's Cypher engine —
        no Python-side computation or full-table load required.  Joins with
        node tables to fetch metadata in a single query.
        """
        assert self._conn is not None
        # Vector literals must be inlined — KuzuDB parameterized queries
        # cannot distinguish DOUBLE[] from LIST for array_cosine_similarity.
        vec_literal = "[" + ", ".join(str(v) for v in vector) + "]"

        try:
            result = self._conn.execute(
                f"MATCH (e:Embedding) "
                f"RETURN e.node_id, "
                f"array_cosine_similarity(e.vec, {vec_literal}) AS sim "
                f"ORDER BY sim DESC LIMIT {limit}"
            )
        except Exception:
            logger.debug("vector_search failed", exc_info=True)
            return []

        emb_rows: list[tuple[str, float]] = []
        while result.has_next():
            row = result.get_next()
            emb_rows.append((row[0] or "", float(row[1]) if row[1] is not None else 0.0))

        if not emb_rows:
            return []

        node_cache: dict[str, GraphNode] = {}
        node_ids = [r[0] for r in emb_rows]
        ids_by_table: dict[str, list[str]] = {}
        for nid in node_ids:
            table = _table_for_id(nid)
            if table:
                ids_by_table.setdefault(table, []).append(nid)

        for table, ids in ids_by_table.items():
            try:
                q = f"MATCH (n:{table}) WHERE n.id IN $ids RETURN n.*"
                res = self._conn.execute(q, parameters={"ids": ids})
                while res.has_next():
                    row = res.get_next()
                    node = self._row_to_node(row)
                    if node:
                        node_cache[node.id] = node
            except Exception:
                logger.debug("Batch node fetch failed for table %s", table, exc_info=True)

        results: list[SearchResult] = []
        for node_id, sim in emb_rows:
            node = node_cache.get(node_id)
            label_prefix = node_id.split(":", 1)[0] if node_id else ""
            results.append(
                SearchResult(
                    node_id=node_id,
                    score=sim,
                    node_name=node.name if node else "",
                    file_path=node.file_path if node else "",
                    label=label_prefix,
                    snippet=(node.content[:200] if node and node.content else ""),
                )
            )
        return results

    def get_indexed_files(self) -> dict[str, str]:
        """Return ``{file_path: sha256(content)}`` for all File nodes.

        Attempts to read pre-computed ``content_hash`` first. Falls back
        to computing the hash from content for databases that predate the
        schema addition.
        """
        assert self._conn is not None
        mapping: dict[str, str] = {}
        try:
            result = self._conn.execute(
                "MATCH (n:File) RETURN n.file_path, n.content"
            )
            while result.has_next():
                row = result.get_next()
                fp = row[0] or ""
                content = row[1] or ""
                mapping[fp] = hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            logger.debug("get_indexed_files failed", exc_info=True)
        return mapping

    def load_graph(self) -> KnowledgeGraph:
        """Reconstruct a full :class:`KnowledgeGraph` from the database."""
        assert self._conn is not None
        graph = KnowledgeGraph()

        # -- Load nodes from every table --
        for table in _NODE_TABLE_NAMES:
            try:
                result = self._conn.execute(f"MATCH (n:{table}) RETURN n.*")
                while result.has_next():
                    row = result.get_next()
                    node = self._row_to_node(row)
                    if node is not None:
                        graph.add_node(node)
            except Exception:
                logger.debug("load_graph: failed to read table %s", table, exc_info=True)

        # -- Load relationships --
        try:
            result = self._conn.execute(
                "MATCH (a)-[r:CodeRelation]->(b) "
                "RETURN a.id, b.id, r.rel_type, r.confidence, r.role, "
                "r.step_number, r.strength, r.co_changes, r.symbols"
            )
            while result.has_next():
                row = result.get_next()
                src_id: str = row[0] or ""
                tgt_id: str = row[1] or ""
                rel_type_str: str = row[2] or ""

                rel_type = _REL_TYPE_MAP.get(rel_type_str)
                if rel_type is None:
                    continue

                rel_id = f"{rel_type_str}:{src_id}->{tgt_id}"

                props: dict[str, Any] = {}
                if row[3] is not None:
                    props["confidence"] = float(row[3])
                if row[4] is not None and row[4] != "":
                    props["role"] = str(row[4])
                if row[5] is not None and row[5] != 0:
                    props["step_number"] = int(row[5])
                if row[6] is not None and row[6] != 0.0:
                    props["strength"] = float(row[6])
                if row[7] is not None and row[7] != 0:
                    props["co_changes"] = int(row[7])
                if row[8] is not None and row[8] != "":
                    props["symbols"] = str(row[8])

                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=rel_type,
                        source=src_id,
                        target=tgt_id,
                        properties=props,
                    )
                )
        except Exception:
            logger.error("load_graph: relationship query failed — graph incomplete", exc_info=True)
            raise

        return graph

    def delete_synthetic_nodes(self) -> None:
        """Remove all COMMUNITY and PROCESS nodes and their relationships."""
        assert self._conn is not None
        for table in ("Community", "Process"):
            try:
                self._conn.execute(f"MATCH (n:{table}) DETACH DELETE n")
            except Exception:
                logger.debug(
                    "delete_synthetic_nodes: failed for %s", table, exc_info=True
                )

    def upsert_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Insert or update embeddings without wiping existing ones."""
        assert self._conn is not None
        for emb in embeddings:
            try:
                self._conn.execute(
                    "MERGE (e:Embedding {node_id: $nid}) SET e.vec = $vec",
                    parameters={"nid": emb.node_id, "vec": emb.embedding},
                )
            except Exception:
                logger.debug(
                    "upsert_embeddings failed for %s", emb.node_id, exc_info=True
                )

    def update_dead_flags(
        self, dead_ids: set[str], alive_ids: set[str]
    ) -> None:
        """Set is_dead=True on *dead_ids* and is_dead=False on *alive_ids*."""
        assert self._conn is not None

        def _batch_set(ids: set[str], value: bool) -> None:
            by_table: dict[str, list[str]] = {}
            for node_id in ids:
                table = _table_for_id(node_id)
                if table:
                    by_table.setdefault(table, []).append(node_id)
            for table, id_list in by_table.items():
                try:
                    self._conn.execute(
                        f"MATCH (n:{table}) WHERE n.id IN $ids SET n.is_dead = $val",
                        parameters={"ids": id_list, "val": value},
                    )
                except Exception:
                    logger.debug(
                        "update_dead_flags failed for table %s", table, exc_info=True
                    )

        _batch_set(dead_ids, True)
        _batch_set(alive_ids, False)

    def remove_relationships_by_type(self, rel_type: RelType) -> None:
        """Delete all relationships of a specific type."""
        assert self._conn is not None
        try:
            self._conn.execute(
                "MATCH ()-[r:CodeRelation]->() WHERE r.rel_type = $rt DELETE r",
                parameters={"rt": rel_type.value},
            )
        except Exception:
            logger.debug(
                "remove_relationships_by_type failed for %s",
                rel_type.value,
                exc_info=True,
            )

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """Replace the entire store with the contents of *graph*.

        Uses CSV-based COPY FROM for bulk loading nodes and relationships,
        falling back to individual inserts if COPY FROM fails.
        """
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            try:
                self._conn.execute(f"MATCH (n:{table}) DETACH DELETE n")
            except Exception:
                pass

        if not self._bulk_load_nodes_csv(graph):
            self.add_nodes(list(graph.iter_nodes()))

        if not self._bulk_load_rels_csv(graph):
            self.add_relationships(list(graph.iter_relationships()))

        self.rebuild_fts_indexes()

    def rebuild_fts_indexes(self) -> None:
        """Drop and recreate all FTS indexes.

        Must be called after any bulk data change so the BM25 indexes
        reflect the current node contents.
        """
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            try:
                self._conn.execute(f"CALL DROP_FTS_INDEX('{table}', '{idx_name}')")
            except Exception:
                pass
            try:
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])"
                )
            except Exception:
                logger.debug("FTS index rebuild failed for %s", table, exc_info=True)

    def _csv_copy(self, table: str, rows: list[list[Any]]) -> None:
        """Write *rows* to a temporary CSV and COPY FROM into *table*.

        Always cleans up the temp file, even on failure.
        """
        assert self._conn is not None
        csv_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                csv_path = f.name
            self._conn.execute(f'COPY {table} FROM "{csv_path}" (HEADER=false)')
        finally:
            if csv_path:
                Path(csv_path).unlink(missing_ok=True)

    def _bulk_load_nodes_csv(self, graph: KnowledgeGraph) -> bool:
        """Load all nodes via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        by_table: dict[str, list[GraphNode]] = {}
        for node in graph.iter_nodes():
            table = _LABEL_TO_TABLE.get(node.label.value)
            if table:
                by_table.setdefault(table, []).append(node)

        try:
            for table, nodes in by_table.items():
                self._csv_copy(table, [
                    [node.id, node.name, node.file_path, node.start_line,
                     node.end_line, node.content, node.signature, node.language,
                     node.class_name, node.is_dead, node.is_entry_point,
                     node.is_exported]
                    for node in nodes
                ])
            return True
        except Exception:
            logger.debug("CSV bulk_load_nodes failed, falling back", exc_info=True)
            return False

    def _bulk_load_rels_csv(self, graph: KnowledgeGraph) -> bool:
        """Load all relationships via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        by_pair: dict[tuple[str, str], list[GraphRelationship]] = {}
        for rel in graph.iter_relationships():
            src_table = _table_for_id(rel.source)
            dst_table = _table_for_id(rel.target)
            if src_table and dst_table:
                by_pair.setdefault((src_table, dst_table), []).append(rel)

        try:
            for (src_table, dst_table), rels in by_pair.items():
                self._csv_copy(f"CodeRelation_{src_table}_{dst_table}", [
                    [rel.source, rel.target, rel.type.value,
                     float((rel.properties or {}).get("confidence", 1.0)),
                     str((rel.properties or {}).get("role", "")),
                     int((rel.properties or {}).get("step_number", 0)),
                     float((rel.properties or {}).get("strength", 0.0)),
                     int((rel.properties or {}).get("co_changes", 0)),
                     str((rel.properties or {}).get("symbols", ""))]
                    for rel in rels
                ])
            return True
        except Exception:
            logger.debug("CSV bulk_load_rels failed, falling back", exc_info=True)
            return False

    def _bulk_store_embeddings_csv(self, embeddings: list[NodeEmbedding]) -> bool:
        """Store embeddings via temporary CSV + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        assert self._conn is not None
        try:
            try:
                self._conn.execute("MATCH (e:Embedding) DETACH DELETE e")
            except Exception:
                pass

            self._csv_copy("Embedding", [
                [emb.node_id,
                 "[" + ",".join(str(v) for v in emb.embedding) + "]"]
                for emb in embeddings
            ])
            return True
        except Exception:
            logger.debug("CSV bulk_store_embeddings failed, falling back", exc_info=True)
            return False

    def _create_schema(self) -> None:
        """Create node/rel/embedding tables and the FTS extension."""
        assert self._conn is not None

        try:
            self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD EXTENSION fts")
        except Exception:
            logger.debug("FTS extension load skipped (may already be loaded)", exc_info=True)

        for table in _NODE_TABLE_NAMES:
            stmt = f"CREATE NODE TABLE IF NOT EXISTS {table}({_NODE_PROPERTIES})"
            self._conn.execute(stmt)

        self._conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS Embedding({_EMBEDDING_PROPERTIES})"
        )

        # Build the REL TABLE GROUP covering all table-to-table combinations.
        from_to_pairs: list[str] = []
        for src in _NODE_TABLE_NAMES:
            for dst in _NODE_TABLE_NAMES:
                from_to_pairs.append(f"FROM {src} TO {dst}")

        pairs_clause = ", ".join(from_to_pairs)
        rel_stmt = (
            f"CREATE REL TABLE GROUP IF NOT EXISTS CodeRelation("
            f"{pairs_clause}, {_REL_PROPERTIES})"
        )
        try:
            self._conn.execute(rel_stmt)
        except Exception:
            logger.debug("REL TABLE GROUP creation skipped", exc_info=True)

        self._create_fts_indexes()

    def _create_fts_indexes(self) -> None:
        """Create FTS indexes for every node table (idempotent)."""
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            try:
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])"
                )
            except Exception:
                # Index may already exist — that's fine.
                pass

    def _insert_node(self, node: GraphNode) -> None:
        """INSERT a single node into the appropriate label table using parameterized query."""
        assert self._conn is not None
        table = _LABEL_TO_TABLE.get(node.label.value)
        if table is None:
            logger.warning("Unknown label %s for node %s", node.label, node.id)
            return

        query = (
            f"CREATE (:{table} {{"
            f"id: $id, name: $name, file_path: $file_path, "
            f"start_line: $start_line, end_line: $end_line, "
            f"content: $content, signature: $signature, "
            f"language: $language, class_name: $class_name, "
            f"is_dead: $is_dead, is_entry_point: $is_entry_point, "
            f"is_exported: $is_exported"
            f"}})"
        )
        params = {
            "id": node.id,
            "name": node.name,
            "file_path": node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "content": node.content,
            "signature": node.signature,
            "language": node.language,
            "class_name": node.class_name,
            "is_dead": node.is_dead,
            "is_entry_point": node.is_entry_point,
            "is_exported": node.is_exported,
        }
        try:
            self._conn.execute(query, parameters=params)
        except Exception:
            logger.debug("Insert node failed for %s", node.id, exc_info=True)

    def _insert_relationship(self, rel: GraphRelationship) -> None:
        """MATCH source and target, then CREATE the relationship using parameterized query."""
        assert self._conn is not None
        src_table = _table_for_id(rel.source)
        tgt_table = _table_for_id(rel.target)
        if src_table is None or tgt_table is None:
            logger.warning(
                "Cannot resolve tables for relationship %s -> %s",
                rel.source,
                rel.target,
            )
            return

        props = rel.properties or {}

        query = (
            f"MATCH (a:{src_table}), (b:{tgt_table}) "
            f"WHERE a.id = $src AND b.id = $tgt "
            f"CREATE (a)-[:CodeRelation {{"
            f"rel_type: $rel_type, "
            f"confidence: $confidence, "
            f"role: $role, "
            f"step_number: $step_number, "
            f"strength: $strength, "
            f"co_changes: $co_changes, "
            f"symbols: $symbols"
            f"}}]->(b)"
        )
        params = {
            "src": rel.source,
            "tgt": rel.target,
            "rel_type": rel.type.value,
            "confidence": float(props.get("confidence", 1.0)),
            "role": str(props.get("role", "")),
            "step_number": int(props.get("step_number", 0)),
            "strength": float(props.get("strength", 0.0)),
            "co_changes": int(props.get("co_changes", 0)),
            "symbols": str(props.get("symbols", "")),
        }
        try:
            self._conn.execute(query, parameters=params)
        except Exception:
            logger.debug(
                "Insert relationship failed: %s -> %s", rel.source, rel.target, exc_info=True
            )

    def _query_nodes(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[GraphNode]:
        """Execute a query returning ``n.*`` columns and convert to GraphNode list."""
        assert self._conn is not None
        nodes: list[GraphNode] = []
        try:
            result = self._conn.execute(query, parameters=parameters or {})
            while result.has_next():
                row = result.get_next()
                node = self._row_to_node(row)
                if node is not None:
                    nodes.append(node)
        except Exception:
            logger.debug("_query_nodes failed: %s", query, exc_info=True)
        return nodes

    def _query_nodes_with_confidence(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[tuple[GraphNode, float]]:
        """Execute a query returning ``n.*`` columns plus a trailing confidence column."""
        assert self._conn is not None
        pairs: list[tuple[GraphNode, float]] = []
        try:
            result = self._conn.execute(query, parameters=parameters or {})
            while result.has_next():
                row = result.get_next()
                node = self._row_to_node(row[:-1])
                confidence = float(row[-1]) if row[-1] is not None else 1.0
                if node is not None:
                    pairs.append((node, confidence))
        except Exception:
            logger.debug("_query_nodes_with_confidence failed: %s", query, exc_info=True)
        return pairs

    @staticmethod
    def _row_to_node(row: list[Any], node_id: str | None = None) -> GraphNode | None:
        """Convert a result row from ``RETURN n.*`` into a GraphNode.

        Column order matches the property definition:
        0=id, 1=name, 2=file_path, 3=start_line, 4=end_line,
        5=content, 6=signature, 7=language, 8=class_name,
        9=is_dead, 10=is_entry_point, 11=is_exported
        """
        try:
            nid = node_id or row[0]
            prefix = nid.split(":", 1)[0]
            label = _LABEL_MAP.get(prefix)
            if label is None:
                logger.warning("Unknown node label prefix %r in id %s", prefix, nid)
                return None

            return GraphNode(
                id=row[0],
                label=label,
                name=row[1] or "",
                file_path=row[2] or "",
                start_line=row[3] or 0,
                end_line=row[4] or 0,
                content=row[5] or "",
                signature=row[6] or "",
                language=row[7] or "",
                class_name=row[8] or "",
                is_dead=bool(row[9]),
                is_entry_point=bool(row[10]),
                is_exported=bool(row[11]),
            )
        except (IndexError, KeyError):
            logger.debug("Failed to convert row to GraphNode: %s", row, exc_info=True)
            return None
