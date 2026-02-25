"""Watch mode for Axon — re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring with
native debouncing.  Changes are processed in tiers:

- **File-local** (immediate): Phases 2-7 on changed files only.
- **Global** (30s batch): Community detection, process detection, dead code.
- **Embeddings** (60s batch): Re-embed changed symbols.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from axon.config.ignore import load_gitignore, should_ignore
from axon.config.languages import is_supported
from axon.core.ingestion.walker import FileEntry, read_file
from axon.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Timer thresholds (seconds).
GLOBAL_PHASE_INTERVAL = 30
EMBEDDING_INTERVAL = 60

def _reindex_files(
    changed_paths: list[Path],
    repo_path: Path,
    storage: StorageBackend,
    gitignore_patterns: list[str] | None = None,
) -> int:
    """Re-index changed files through file-local phases.

    Filters out ignored and unsupported files, reads them, then runs
    the mini-pipeline via ``reindex_files()``.

    Returns the number of files actually reindexed.
    """
    from axon.core.ingestion.pipeline import reindex_files

    entries: list[FileEntry] = []
    for abs_path in changed_paths:
        if not abs_path.is_file():
            # File was deleted — remove from storage.
            try:
                relative = str(abs_path.relative_to(repo_path))
                storage.remove_nodes_by_file(relative)
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

        # RACE-1: Handle file disappearing between is_file() check and read.
        try:
            entry = read_file(repo_path, abs_path)
        except (FileNotFoundError, PermissionError, OSError):
            logger.debug("File disappeared or became inaccessible: %s", abs_path)
            continue
        if entry is not None:
            entries.append(entry)

    if entries:
        reindex_files(entries, repo_path, storage)

    return len(entries)

def _run_global_phases(storage: StorageBackend, repo_path: Path) -> None:
    """Run global analysis phases (communities, processes, dead code).

    Rebuilds the full in-memory graph from storage is not practical for
    incremental mode, so we run a full pipeline refresh.  In practice this
    is fast because the storage is already populated.
    """
    from axon.core.ingestion.pipeline import run_pipeline

    run_pipeline(repo_path, storage=storage, full=True)
    logger.info("Global phases completed")

async def watch_repo(
    repo_path: Path,
    storage: StorageBackend,
    *,
    stop_event: asyncio.Event | None = None,
    lock: asyncio.Lock | None = None,
) -> None:
    """Main watch loop — monitor files and re-index on changes.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to watch.
    storage:
        An already-initialised storage backend.
    stop_event:
        Optional event to signal shutdown (useful for testing).
        When set, the watch loop exits gracefully.
    lock:
        Optional async lock for coordinating storage access with
        concurrent readers (e.g. the MCP server in combined mode).
    """
    import watchfiles

    async def _run_sync(fn, *args):
        """Run a sync function in a thread, optionally under the lock."""
        if lock is not None:
            async with lock:
                return await asyncio.to_thread(fn, *args)
        return await asyncio.to_thread(fn, *args)

    gitignore = load_gitignore(repo_path)
    dirty = False
    last_global = time.monotonic()
    files_changed = 0

    logger.info("Watching %s for changes...", repo_path)

    async for changes in watchfiles.awatch(
        repo_path,
        rust_timeout=500,
        stop_event=stop_event,
    ):
        changed_paths: list[Path] = []
        seen: set[str] = set()
        for _change_type, path_str in changes:
            if path_str not in seen:
                seen.add(path_str)
                changed_paths.append(Path(path_str))

        if not changed_paths:
            continue

        count = await _run_sync(_reindex_files, changed_paths, repo_path, storage, gitignore)
        if count > 0:
            files_changed += count
            dirty = True
            logger.info("Reindexed %d file(s)", count)

        now = time.monotonic()
        if dirty and (now - last_global) >= GLOBAL_PHASE_INTERVAL:
            logger.info("Running global analysis phases...")
            await _run_sync(_run_global_phases, storage, repo_path)
            dirty = False
            last_global = now

    logger.info("Watch stopped. Total files reindexed: %d", files_changed)
