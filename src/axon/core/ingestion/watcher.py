"""Watch mode for Axon — re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring with
native debouncing.  Changes are processed in tiers:

- **File-local** (immediate): Phases 2-7 on changed files only.
- **Global** (after quiet period): Hydrate graph from storage, run
  communities/processes/dead-code on the full graph.
- **Embeddings** (with global): Re-embed dirty nodes + CALLS neighbors.
- **Coupling** (commit-triggered): Re-run git coupling when HEAD changes.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path

from axon.config.ignore import load_gitignore, should_ignore
from axon.config.languages import is_supported
from axon.core.ingestion.walker import FileEntry, read_file
from axon.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Debounce: global phases fire after this many seconds of no file changes.
QUIET_PERIOD = 5.0

# How often watchfiles yields (controls quiet-period check granularity).
_POLL_INTERVAL_MS = 500


def _get_head_sha(repo_path: Path) -> str | None:
    """Return the current git HEAD sha, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _reindex_files(
    changed_paths: list[Path],
    repo_path: Path,
    storage: StorageBackend,
    gitignore_patterns: list[str] | None = None,
) -> tuple[int, set[str]]:
    """Re-index changed files through file-local phases.

    Returns (count_reindexed, set_of_relative_file_paths_reindexed).
    """
    from axon.core.ingestion.pipeline import reindex_files

    entries: list[FileEntry] = []
    reindexed_paths: set[str] = set()

    for abs_path in changed_paths:
        if not abs_path.is_file():
            try:
                relative = str(abs_path.relative_to(repo_path))
                storage.remove_nodes_by_file(relative)
                reindexed_paths.add(relative)
            except (ValueError, OSError):
                pass
            continue

        try:
            relative = str(abs_path.relative_to(repo_path))
        except ValueError:
            continue

        if should_ignore(relative, gitignore_patterns):
            continue

        if not is_supported(abs_path):
            continue

        entry = read_file(repo_path, abs_path)
        if entry is not None:
            entries.append(entry)
            reindexed_paths.add(relative)

    if entries:
        reindex_files(entries, repo_path, storage)

    return len(entries), reindexed_paths


def _compute_dirty_node_ids(storage: StorageBackend, dirty_files: set[str]) -> set[str]:
    """Find all node IDs in dirty files + their immediate CALLS neighbors."""
    if not dirty_files:
        return set()

    dirty_node_ids: set[str] = set()
    for file_path in dirty_files:
        escaped = file_path.replace("\\", "\\\\").replace("'", "\\'")
        results = storage.execute_raw(
            f"MATCH (n) WHERE n.file_path = '{escaped}' RETURN n.id"
        )
        for row in results:
            if row[0]:
                dirty_node_ids.add(row[0])

    neighbor_ids: set[str] = set()
    for node_id in dirty_node_ids:
        for caller in storage.get_callers(node_id):
            neighbor_ids.add(caller.id)
        for callee in storage.get_callees(node_id):
            neighbor_ids.add(callee.id)

    return dirty_node_ids | neighbor_ids


def _run_incremental_global_phases(
    storage: StorageBackend,
    repo_path: Path,
    dirty_files: set[str],
    run_coupling: bool = False,
) -> None:
    """Run global phases incrementally using graph hydrated from storage."""
    from axon.core.ingestion.community import process_communities
    from axon.core.ingestion.coupling import process_coupling
    from axon.core.ingestion.dead_code import process_dead_code
    from axon.core.ingestion.processes import process_processes
    from axon.core.embeddings.embedder import embed_nodes
    from axon.core.graph.model import NodeLabel, RelType

    logger.info("Hydrating graph from storage...")
    graph = storage.load_graph()

    # --- Communities ---
    storage.delete_synthetic_nodes()
    num_communities = process_communities(graph)
    logger.info("Communities: %d", num_communities)

    # --- Processes (depends on communities for kind classification) ---
    num_processes = process_processes(graph)
    logger.info("Processes: %d", num_processes)

    # --- Dead code ---
    num_dead = process_dead_code(graph)
    logger.info("Dead code: %d", num_dead)

    # Persist new synthetic nodes and relationships.
    new_nodes = (
        list(graph.get_nodes_by_label(NodeLabel.COMMUNITY))
        + list(graph.get_nodes_by_label(NodeLabel.PROCESS))
    )
    new_rels = (
        list(graph.get_relationships_by_type(RelType.MEMBER_OF))
        + list(graph.get_relationships_by_type(RelType.STEP_IN_PROCESS))
    )
    if new_nodes:
        storage.add_nodes(new_nodes)
    if new_rels:
        storage.add_relationships(new_rels)

    dead_ids: set[str] = set()
    alive_ids: set[str] = set()
    _SYMBOL_LABELS = (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)
    for label in _SYMBOL_LABELS:
        for node in graph.get_nodes_by_label(label):
            if node.is_dead:
                dead_ids.add(node.id)
            else:
                alive_ids.add(node.id)
    storage.update_dead_flags(dead_ids, alive_ids)

    # --- Coupling (only if new commits) ---
    if run_coupling:
        storage.remove_relationships_by_type(RelType.COUPLED_WITH)
        num_coupled = process_coupling(graph, repo_path)
        coupled_rels = list(graph.get_relationships_by_type(RelType.COUPLED_WITH))
        if coupled_rels:
            storage.add_relationships(coupled_rels)
        logger.info("Coupling: %d pairs", num_coupled)

    # --- Incremental embeddings ---
    dirty_node_ids = _compute_dirty_node_ids(storage, dirty_files)
    if dirty_node_ids:
        logger.info("Re-embedding %d nodes...", len(dirty_node_ids))
        try:
            embeddings = embed_nodes(graph, dirty_node_ids)
            if embeddings:
                storage.upsert_embeddings(embeddings)
        except Exception:
            logger.warning("Incremental embedding failed", exc_info=True)

    storage.rebuild_fts_indexes()

    logger.info("Incremental global phases complete.")


async def watch_repo(
    repo_path: Path,
    storage: StorageBackend,
    *,
    stop_event: asyncio.Event | None = None,
    lock: asyncio.Lock | None = None,
) -> None:
    """Main watch loop — monitor files and re-index on changes.

    File-local reindex runs immediately on every change. Global phases
    (communities, processes, dead code, embeddings) run after a quiet
    period of QUIET_PERIOD seconds with no new changes. Coupling runs
    only when new git commits are detected.
    """
    import watchfiles

    async def _run_sync(fn, *args):
        if lock is not None:
            async with lock:
                return await asyncio.to_thread(fn, *args)
        return await asyncio.to_thread(fn, *args)

    gitignore = load_gitignore(repo_path)
    dirty_files: set[str] = set()
    last_change_time: float = 0.0
    global_running = False
    last_known_commit = _get_head_sha(repo_path)

    logger.info("Watching %s for changes...", repo_path)

    async for changes in watchfiles.awatch(
        repo_path,
        rust_timeout=_POLL_INTERVAL_MS,
        stop_event=stop_event,
    ):
        # --- Tier 1: Immediate file-local reindex ---
        changed_paths: list[Path] = []
        seen: set[str] = set()
        for _change_type, path_str in changes:
            if path_str not in seen:
                seen.add(path_str)
                changed_paths.append(Path(path_str))

        if changed_paths:
            count, reindexed = await _run_sync(
                _reindex_files, changed_paths, repo_path, storage, gitignore,
            )
            if reindexed:
                dirty_files |= reindexed
                last_change_time = time.monotonic()
                logger.info("Reindexed %d file(s), %d paths dirty", count, len(reindexed))

        # --- Tier 2: Debounced global phases ---
        now = time.monotonic()
        if (
            dirty_files
            and not global_running
            and last_change_time > 0
            and (now - last_change_time) >= QUIET_PERIOD
        ):
            global_running = True
            snapshot = dirty_files.copy()
            dirty_files.clear()

            current_commit = _get_head_sha(repo_path)
            run_coupling = current_commit != last_known_commit
            if run_coupling:
                last_known_commit = current_commit

            logger.info("Running incremental global phases...")
            await _run_sync(
                _run_incremental_global_phases,
                storage, repo_path, snapshot, run_coupling,
            )
            global_running = False

    logger.info("Watch stopped.")
