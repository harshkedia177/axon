"""MCP tool handler implementations for Axon.

Each function accepts a storage backend and the tool-specific arguments,
performs the appropriate query, and returns a human-readable string suitable
for inclusion in an MCP ``TextContent`` response.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from axon.core.search.hybrid import hybrid_search
from axon.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

MAX_TRAVERSE_DEPTH = 10

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _confidence_tag(confidence: float) -> str:
    """Return a visual confidence indicator for edge display."""
    if confidence >= 0.9:
        return ""
    if confidence >= 0.5:
        return " (~)"
    return " (?)"


def _resolve_symbol(storage: StorageBackend, symbol: str) -> list:
    """Resolve a symbol name to search results, preferring exact name matches."""
    if hasattr(storage, "exact_name_search"):
        results = storage.exact_name_search(symbol, limit=1)
        if results:
            return results
    return storage.fts_search(symbol, limit=1)

def handle_list_repos(registry_dir: Path | None = None) -> str:
    """List indexed repositories by scanning for .axon directories.

    Scans the global registry directory (defaults to ``~/.axon/repos``) for
    project metadata files and returns a formatted summary.

    Args:
        registry_dir: Directory containing repo metadata. If ``None``,
            defaults to ``~/.axon/repos``.

    Returns:
        Formatted list of indexed repositories with stats, or a message
        indicating none were found.
    """
    use_cwd_fallback = registry_dir is None
    if registry_dir is None:
        registry_dir = Path.home() / ".axon" / "repos"

    repos: list[dict[str, Any]] = []

    if registry_dir.exists():
        for meta_file in registry_dir.glob("*/meta.json"):
            try:
                data = json.loads(meta_file.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    if not repos and use_cwd_fallback:
        # Fall back: scan current directory for .axon
        cwd_axon = Path.cwd() / ".axon" / "meta.json"
        if cwd_axon.exists():
            try:
                data = json.loads(cwd_axon.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                pass

    if not repos:
        return "No indexed repositories found. Run `axon index` on a project first."

    lines = [f"Indexed repositories ({len(repos)}):"]
    lines.append("")
    for i, repo in enumerate(repos, 1):
        name = repo.get("name", "unknown")
        path = repo.get("path", "")
        stats = repo.get("stats", {})
        files = stats.get("files", "?")
        symbols = stats.get("symbols", "?")
        relationships = stats.get("relationships", "?")
        lines.append(f"  {i}. {name}")
        lines.append(f"     Path: {path}")
        lines.append(f"     Files: {files}  Symbols: {symbols}  Relationships: {relationships}")
        lines.append("")

    return "\n".join(lines)

def _group_by_process(
    results: list,
    storage: StorageBackend,
) -> dict[str, list]:
    """Map search results to their parent execution processes.

    Delegates to ``storage.get_process_memberships()`` for a safe
    parameterized query, falling back to an empty dict if the backend
    does not support the method.
    """
    if not results:
        return {}

    node_ids = [r.node_id for r in results]

    try:
        node_to_process = storage.get_process_memberships(node_ids)
    except (AttributeError, TypeError):
        return {}

    groups: dict[str, list] = {}
    for r in results:
        pname = node_to_process.get(r.node_id)
        if pname:
            groups.setdefault(pname, []).append(r)

    return groups


def _format_query_results(results: list, groups: dict[str, list]) -> str:
    """Format search results with process grouping.

    Results belonging to a process appear under a labelled section.
    Remaining results appear in an "Other results" section.
    """
    grouped_ids: set[str] = {r.node_id for group in groups.values() for r in group}
    ungrouped = [r for r in results if r.node_id not in grouped_ids]

    lines: list[str] = []
    counter = 1

    for process_name, proc_results in groups.items():
        lines.append(f"=== {process_name} ===")
        for r in proc_results:
            label = r.label.title() if r.label else "Unknown"
            lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
            if r.snippet:
                snippet = r.snippet[:200].replace("\n", " ").strip()
                lines.append(f"   {snippet}")
            counter += 1
        lines.append("")

    if ungrouped:
        if groups:
            lines.append("=== Other results ===")
        for r in ungrouped:
            label = r.label.title() if r.label else "Unknown"
            lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
            if r.snippet:
                snippet = r.snippet[:200].replace("\n", " ").strip()
                lines.append(f"   {snippet}")
            counter += 1
        lines.append("")

    lines.append("Next: Use context() on a specific symbol for the full picture.")
    return "\n".join(lines)


def handle_query(storage: StorageBackend, query: str, limit: int = 20) -> str:
    """Execute hybrid search and format results, grouped by execution process.

    Args:
        storage: The storage backend to search against.
        query: Text search query.
        limit: Maximum number of results (default 20).

    Returns:
        Formatted search results grouped by process, with file, name, label,
        and snippet for each result.
    """
    query_embedding: list[float] | None = None
    try:
        from axon.core.embeddings.embedder import _get_model

        model = _get_model(_EMBED_MODEL_NAME)
        query_embedding = list(next(iter(model.embed([query]))))
    except Exception:
        logger.debug("Query embedding failed, falling back to FTS only", exc_info=True)

    results = hybrid_search(query, storage, query_embedding=query_embedding, limit=limit)
    if not results:
        return f"No results found for '{query}'."

    groups = _group_by_process(results, storage)
    return _format_query_results(results, groups)

def handle_context(storage: StorageBackend, symbol: str) -> str:
    """Provide a 360-degree view of a symbol.

    Looks up the symbol by name via full-text search, then retrieves its
    callers, callees, and type references.

    Args:
        storage: The storage backend.
        symbol: The symbol name to look up.

    Returns:
        Formatted view including callers, callees, type refs, and guidance.
    """
    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        return f"Symbol '{symbol}' not found."

    label_display = node.label.value.title() if node.label else "Unknown"
    lines = [f"Symbol: {node.name} ({label_display})"]
    lines.append(f"File: {node.file_path}:{node.start_line}-{node.end_line}")

    if node.signature:
        lines.append(f"Signature: {node.signature}")

    if node.is_dead:
        lines.append("Status: DEAD CODE (unreachable)")

    try:
        callers_raw = storage.get_callers_with_confidence(node.id)
    except (AttributeError, TypeError):
        callers_raw = [(c, 1.0) for c in storage.get_callers(node.id)]

    if callers_raw:
        lines.append(f"\nCallers ({len(callers_raw)}):")
        for c, conf in callers_raw:
            tag = _confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

    try:
        callees_raw = storage.get_callees_with_confidence(node.id)
    except (AttributeError, TypeError):
        callees_raw = [(c, 1.0) for c in storage.get_callees(node.id)]

    if callees_raw:
        lines.append(f"\nCallees ({len(callees_raw)}):")
        for c, conf in callees_raw:
            tag = _confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

    type_refs = storage.get_type_refs(node.id)
    if type_refs:
        lines.append(f"\nType references ({len(type_refs)}):")
        for t in type_refs:
            lines.append(f"  -> {t.name}  {t.file_path}")

    lines.append("")
    lines.append("Next: Use impact() if planning changes to this symbol.")
    return "\n".join(lines)

_DEPTH_LABELS: dict[int, str] = {
    1: "Direct callers (will break)",
    2: "Indirect (may break)",
}


def handle_impact(storage: StorageBackend, symbol: str, depth: int = 3) -> str:
    """Analyse the blast radius of changing a symbol, grouped by hop depth.

    Uses BFS traversal through CALLS edges to find all affected symbols
    up to the specified depth, then groups results by distance.

    Args:
        storage: The storage backend.
        symbol: The symbol name to analyse.
        depth: Maximum traversal depth (default 3).

    Returns:
        Formatted impact analysis with depth-grouped sections.
    """
    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        return f"Symbol '{symbol}' not found."

    affected_with_depth = storage.traverse_with_depth(
        start_node.id, depth, direction="callers"
    )
    if not affected_with_depth:
        return f"No upstream callers found for '{symbol}'."

    # Group by depth
    by_depth: dict[int, list] = {}
    for node, d in affected_with_depth:
        by_depth.setdefault(d, []).append(node)

    total = len(affected_with_depth)
    label_display = start_node.label.value.title()
    lines = [f"Impact analysis for: {start_node.name} ({label_display})"]
    lines.append(f"Depth: {depth} | Total: {total} symbols")

    # Build confidence lookup for depth-1 (direct callers) display
    conf_lookup: dict[str, float] = {}
    try:
        for node, conf in storage.get_callers_with_confidence(start_node.id):
            conf_lookup[node.id] = conf
    except (AttributeError, TypeError):
        pass

    counter = 1
    for d in sorted(by_depth.keys()):
        depth_label = _DEPTH_LABELS.get(d, "Transitive (review)")
        lines.append(f"\nDepth {d} — {depth_label}:")
        for node in by_depth[d]:
            label = node.label.value.title() if node.label else "Unknown"
            conf = conf_lookup.get(node.id)
            tag = f"  (confidence: {conf:.2f})" if conf is not None else ""
            lines.append(
                f"  {counter}. {node.name} ({label}) -- "
                f"{node.file_path}:{node.start_line}{tag}"
            )
            counter += 1

    lines.append("")
    lines.append("Tip: Review each affected symbol before making changes.")
    return "\n".join(lines)

def handle_dead_code(storage: StorageBackend) -> str:
    """List all symbols marked as dead code.

    Delegates to :func:`~axon.mcp.resources.get_dead_code_list` for the
    shared query and formatting.

    Args:
        storage: The storage backend.

    Returns:
        Formatted list of dead code symbols grouped by file.
    """
    from axon.mcp.resources import get_dead_code_list

    return get_dead_code_list(storage)

_DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_DIFF_HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

def handle_detect_changes(storage: StorageBackend, diff: str) -> str:
    """Map git diff output to affected symbols.

    Parses the diff to find changed files and line ranges, then queries
    the storage backend to identify which symbols those lines belong to.

    Args:
        storage: The storage backend.
        diff: Raw git diff output string.

    Returns:
        Formatted list of affected symbols per changed file.
    """
    if not diff.strip():
        return "Empty diff provided."

    changed_files: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split("\n"):
        file_match = _DIFF_FILE_PATTERN.match(line)
        if file_match:
            current_file = file_match.group(2)
            if current_file not in changed_files:
                changed_files[current_file] = []
            continue

        hunk_match = _DIFF_HUNK_PATTERN.match(line)
        if hunk_match and current_file is not None:
            start = int(hunk_match.group(1))
            count = int(hunk_match.group(2) or "1")
            changed_files[current_file].append((start, start + count - 1))

    if not changed_files:
        return "Could not parse any changed files from the diff."

    lines = [f"Changed files: {len(changed_files)}"]
    lines.append("")
    total_affected = 0

    for file_path, ranges in changed_files.items():
        affected_symbols = []
        try:
            # INJ-1: Use parameterized query_symbols_by_file instead of
            # f-string interpolation into execute_raw.
            rows = storage.query_symbols_by_file(file_path)
            for row in rows or []:
                node_id = row[0] or ""
                name = row[1] or ""
                start_line = row[3] or 0
                end_line = row[4] or 0
                label_prefix = node_id.split(":", 1)[0] if node_id else ""
                for start, end in ranges:
                    if start_line <= end and end_line >= start:
                        affected_symbols.append(
                            (name, label_prefix.title(), start_line, end_line)
                        )
                        break
        except Exception as exc:
            logger.warning("Failed to query symbols for %s: %s", file_path, exc, exc_info=True)
            lines.append(f"  {file_path}:")
            # ERR-1: omit exc from user-facing message to avoid leaking internals
            lines.append("    (error querying symbols)")
            lines.append("")
            continue

        lines.append(f"  {file_path}:")
        if affected_symbols:
            for sym_name, label, s_line, e_line in affected_symbols:
                lines.append(f"    - {sym_name} ({label}) lines {s_line}-{e_line}")
                total_affected += 1
        else:
            lines.append("    (no indexed symbols in changed lines)")
        lines.append("")

    lines.append(f"Total affected symbols: {total_affected}")
    lines.append("")
    lines.append("Next: Use impact() on affected symbols to see downstream effects.")
    return "\n".join(lines)


def handle_cypher(storage: StorageBackend, query: str) -> str:
    """Execute a raw Cypher query and return formatted results.

    Only read-only queries are allowed.  The database is opened in
    read-only mode, so the engine itself will reject write operations.

    Args:
        storage: The storage backend.
        query: The Cypher query string.

    Returns:
        Formatted query results, or an error message if execution fails.
    """
    # INJ-4: Removed regex-based _WRITE_KEYWORDS denylist. The database
    # connection is opened in read-only mode (server.py), so KuzuDB
    # enforces read-only at the engine level.
    try:
        rows = storage.execute_read_query(query)
    except Exception as exc:
        err_msg = str(exc).lower()
        # Provide a user-friendly message if the DB rejects a write attempt.
        if "read" in err_msg and "only" in err_msg:
            return (
                "Query rejected: the database is open in read-only mode. "
                "Write operations (DELETE, DROP, CREATE, SET, MERGE) are not permitted."
            )
        # ERR-1: keep full details in logs only; return generic message to client
        logger.warning("Cypher query failed: %s", exc, exc_info=True)
        return "Cypher query failed. Check server logs for details."

    if not rows:
        return "Query returned no results."

    lines = [f"Results ({len(rows)} rows):"]
    lines.append("")
    for i, row in enumerate(rows, 1):
        formatted_values = [str(v) for v in row]
        lines.append(f"  {i}. {' | '.join(formatted_values)}")

    return "\n".join(lines)
