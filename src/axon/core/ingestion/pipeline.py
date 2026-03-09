"""Pipeline orchestrator for Axon.

Runs all ingestion phases in sequence, populates an in-memory knowledge graph,
bulk-loads it into a storage backend, and returns a summary of the results.

Phases executed:
    0. Incremental diff (reserved -- not yet implemented)
    1. File walking
    2. Structure processing (File/Folder nodes + CONTAINS edges)
    3. Code parsing (symbol nodes + DEFINES edges)
    4. Import resolution (IMPORTS edges)
    5. Call tracing (CALLS edges)
    6. Heritage extraction (EXTENDS / IMPLEMENTS edges)
    7. Type analysis (USES_TYPE edges)
    8. Community detection (COMMUNITY nodes + MEMBER_OF edges)
    9. Process detection (PROCESS nodes + STEP_IN_PROCESS edges)
    10. Dead code detection (flags unreachable symbols)
    11. Change coupling (COUPLED_WITH edges from git history)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from axon.config.ignore import load_gitignore
from axon.core.embeddings.embedder import embed_graph
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphRelationship, NodeLabel
from axon.core.ingestion.calls import process_calls
from axon.core.ingestion.community import process_communities
from axon.core.ingestion.coupling import resolve_coupling
from axon.core.ingestion.dead_code import process_dead_code
from axon.core.ingestion.heritage import process_heritage
from axon.core.ingestion.imports import build_file_index, process_imports
from axon.core.ingestion.parser_phase import process_parsing
from axon.core.ingestion.processes import process_processes
from axon.core.ingestion.resolved import ResolvedEdge
from axon.core.ingestion.structure import process_structure
from axon.core.ingestion.symbol_lookup import build_name_index
from axon.core.ingestion.types import process_types
from axon.core.ingestion.walker import FileEntry, walk_repo
from axon.core.storage.base import StorageBackend


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    files: int = 0
    symbols: int = 0
    relationships: int = 0
    clusters: int = 0
    processes: int = 0
    dead_code: int = 0
    coupled_pairs: int = 0
    embeddings: int = 0
    duration_seconds: float = 0.0
    incremental: bool = False
    changed_files: int = 0

_SYMBOL_LABELS: frozenset[NodeLabel] = frozenset(NodeLabel) - {
    NodeLabel.FILE,
    NodeLabel.FOLDER,
    NodeLabel.COMMUNITY,
    NodeLabel.PROCESS,
}

def _write_collected_edges(
    edges: list[ResolvedEdge],
    graph: KnowledgeGraph,
) -> None:
    """Write a batch of resolved edges to the graph (sequential, deduped)."""
    for edge in edges:
        graph.add_relationship(
            GraphRelationship(
                id=edge.rel_id,
                type=edge.rel_type,
                source=edge.source,
                target=edge.target,
                properties=edge.properties,
            )
        )


def run_pipeline(
    repo_path: Path,
    storage: StorageBackend | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    embeddings: bool = True,
) -> tuple[KnowledgeGraph, PipelineResult]:
    """Run phases 1-11 of the ingestion pipeline.

    When *storage* is provided the graph is bulk-loaded into it after
    all phases complete.  When ``None``, only the in-memory graph is
    returned (useful for branch comparison snapshots).

    Parameters
    ----------
    repo_path:
        Root directory of the repository to analyse.
    storage:
        An already-initialised :class:`StorageBackend` to persist the graph.
        Pass ``None`` to skip storage loading.
    progress_callback:
        Optional ``(phase_name, progress)`` callback where *progress* is a
        float in ``[0.0, 1.0]``.
    embeddings:
        When ``True`` (default), generate and store vector embeddings after
        bulk-loading.  Set to ``False`` to skip embedding generation.

    Returns
    -------
    tuple[KnowledgeGraph, PipelineResult]
        The populated graph and a summary dataclass with counts and timings.
    """
    start = time.monotonic()
    result = PipelineResult()
    log = logging.getLogger(__name__)

    phase_times: dict[str, float] = {}

    def report(phase: str, pct: float) -> None:
        if progress_callback is not None:
            progress_callback(phase, pct)

    @contextmanager
    def _timed(phase_name: str):
        """Context manager that logs and records phase wall-clock time."""
        report(phase_name, 0.0)
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            phase_times[phase_name] = elapsed
            log.info("Phase %-30s  %.2fs", phase_name, elapsed)
            report(phase_name, 1.0)

    with _timed("Walking files"):
        gitignore = load_gitignore(repo_path)
        files = walk_repo(repo_path, gitignore)
        result.files = len(files)

    graph = KnowledgeGraph()

    with _timed("Processing structure"):
        process_structure(files, graph)

    with _timed("Parsing code"):
        parse_data = process_parsing(files, graph)

    with _timed("Resolving imports"):
        process_imports(parse_data, graph, parallel=True)

    with _timed("Building indexes"):
        shared_labels = (
            NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS,
            NodeLabel.INTERFACE, NodeLabel.TYPE_ALIAS,
        )
        shared_name_index = build_name_index(graph, shared_labels)
        heritage_labels = {NodeLabel.CLASS, NodeLabel.INTERFACE}
        heritage_name_index: dict[str, list[str]] = {}
        for name, ids in shared_name_index.items():
            filtered = [
                nid for nid in ids
                if (n := graph.get_node(nid)) is not None and n.label in heritage_labels
            ]
            if filtered:
                heritage_name_index[name] = filtered

    with _timed("Resolving relationships"):
        with ThreadPoolExecutor(max_workers=3) as pool:
            calls_f = pool.submit(
                process_calls, parse_data, graph,
                name_index=shared_name_index, parallel=False, collect=True,
            )
            heritage_f = pool.submit(
                process_heritage, parse_data, graph,
                name_index=heritage_name_index, parallel=False, collect=True,
            )
            types_f = pool.submit(
                process_types, parse_data, graph,
                name_index=shared_name_index, parallel=False, collect=True,
            )

        _write_collected_edges(calls_f.result() or [], graph)

        heritage_edges, heritage_patches = heritage_f.result()
        _write_collected_edges(heritage_edges, graph)
        for patch in heritage_patches:
            node = graph.get_node(patch.node_id)
            if node is not None:
                node.properties[patch.key] = patch.value

        _write_collected_edges(types_f.result() or [], graph)

    coupling_file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)

    with ThreadPoolExecutor(max_workers=1) as pool:
        coupling_future = pool.submit(
            resolve_coupling, graph, repo_path, file_nodes=coupling_file_nodes,
        )

        with _timed("Detecting communities"):
            result.clusters = process_communities(graph)

        with _timed("Detecting execution flows"):
            result.processes = process_processes(graph)

        with _timed("Finding dead code"):
            result.dead_code = process_dead_code(graph)

        with _timed("Analyzing git history"):
            coupling_edges = coupling_future.result()
            _write_collected_edges(coupling_edges, graph)
            result.coupled_pairs = len(coupling_edges)

    result.symbols = sum(1 for n in graph.iter_nodes() if n.label in _SYMBOL_LABELS)
    result.relationships = graph.relationship_count

    if storage is not None:
        with _timed("Loading to storage"):
            storage.bulk_load(graph)

        if embeddings:
            try:
                with _timed("Generating embeddings"):
                    node_embeddings = embed_graph(graph)
                    storage.store_embeddings(node_embeddings)
                    result.embeddings = len(node_embeddings)
            except Exception:
                log.warning(
                    "Embedding phase failed — search will use FTS only",
                    exc_info=True,
                )
                report("Generating embeddings", 1.0)

    result.duration_seconds = time.monotonic() - start

    # Log phase breakdown summary
    if phase_times:
        log.info("─── Phase timing breakdown ───")
        for phase, elapsed in phase_times.items():
            pct = (elapsed / result.duration_seconds * 100) if result.duration_seconds > 0 else 0
            log.info("  %-30s %6.1fs  (%4.1f%%)", phase, elapsed, pct)
        log.info("  %-30s %6.1fs", "TOTAL", result.duration_seconds)

    return graph, result

def reindex_files(
    file_entries: list[FileEntry],
    repo_path: Path,
    storage: StorageBackend,
    rebuild_fts: bool = True,
) -> KnowledgeGraph:
    """Re-index specific files through phases 2-7 (file-local phases).

    Removes old nodes for these files from storage, re-parses them,
    and inserts updated nodes/relationships. Returns the partial graph
    for further processing (global phases, embeddings).

    Note: This only re-runs file-local phases (structure through types).
    Global phases (communities, processes, dead code, coupling) are NOT
    re-run. The caller must invoke these separately or use ``watch_repo``
    which handles this automatically.

    Parameters
    ----------
    file_entries:
        The files to re-index (already read from disk).
    repo_path:
        Root directory of the repository.
    storage:
        An already-initialised storage backend.

    Returns
    -------
    KnowledgeGraph
        The hydrated in-memory graph used for incremental resolution.
    """
    changed_files = {entry.path for entry in file_entries}
    saved_edges: list[GraphRelationship] = []
    for fp in changed_files:
        saved_edges.extend(
            storage.get_inbound_cross_file_edges(fp, exclude_source_files=changed_files)
        )

    for entry in file_entries:
        storage.remove_nodes_by_file(entry.path)

    graph = storage.load_graph()

    process_structure(file_entries, graph)
    parse_data = process_parsing(file_entries, graph)

    file_index = build_file_index(graph)

    shared_labels = (
        NodeLabel.FUNCTION,
        NodeLabel.METHOD,
        NodeLabel.CLASS,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
    )
    shared_name_index = build_name_index(graph, shared_labels)
    heritage_labels = {NodeLabel.CLASS, NodeLabel.INTERFACE}
    heritage_name_index: dict[str, list[str]] = {}
    for name, ids in shared_name_index.items():
        filtered = [
            nid for nid in ids
            if (n := graph.get_node(nid)) is not None and n.label in heritage_labels
        ]
        if filtered:
            heritage_name_index[name] = filtered

    process_imports(parse_data, graph, file_index=file_index)
    process_calls(parse_data, graph, name_index=shared_name_index)
    process_heritage(parse_data, graph, name_index=heritage_name_index)
    process_types(parse_data, graph, name_index=shared_name_index)

    incremental_nodes = [
        node for node in graph.iter_nodes()
        if (
            node.file_path in changed_files
            or (
                node.label == NodeLabel.FOLDER
                and any(
                    file_path == node.file_path or file_path.startswith(f"{node.file_path}/")
                    for file_path in changed_files
                )
            )
        )
    ]
    incremental_node_ids = {node.id for node in incremental_nodes}
    incremental_relationships = [
        rel for rel in graph.iter_relationships()
        if rel.source in incremental_node_ids or rel.target in incremental_node_ids
    ]

    storage.add_nodes(incremental_nodes)
    storage.add_relationships(incremental_relationships)

    if saved_edges:
        storage.add_relationships(saved_edges)

    if rebuild_fts:
        storage.rebuild_fts_indexes()

    return graph

def build_graph(repo_path: Path) -> KnowledgeGraph:
    """Run phases 1-11 and return the in-memory graph without persisting to storage."""
    graph, _ = run_pipeline(repo_path, embeddings=False)
    return graph
