"""Shared symbol lookup utilities for ingestion phases.

Provides efficient line-based containment lookups using a pre-built
per-file interval index, replacing the O(N) scan approach with
O(log N) binary search.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel


def build_name_index(
    graph: KnowledgeGraph,
    labels: tuple[NodeLabel, ...],
) -> dict[str, list[str]]:
    """Build a mapping from symbol names to their node IDs.

    Iterates over all nodes matching the given *labels* and groups
    them by name.  Multiple symbols can share the same name across
    different files, so each entry maps to a list of node IDs.

    This is the shared implementation used by calls, heritage, and
    type analysis phases.
    """
    index: dict[str, list[str]] = {}
    for label in labels:
        for node in graph.get_nodes_by_label(label):
            index.setdefault(node.name, []).append(node.id)
    return index


class FileSymbolIndex:
    """Pre-built per-file interval index for fast containment lookups.

    Stores ``(start_line, end_line, span, node_id)`` tuples sorted by
    ``start_line`` alongside a pre-computed ``start_lines`` list for
    O(log N) binary search without per-lookup list creation.
    """

    __slots__ = ("_entries", "_start_lines")

    def __init__(
        self,
        entries: dict[str, list[tuple[int, int, int, str]]],
        start_lines: dict[str, list[int]],
    ) -> None:
        self._entries = entries
        self._start_lines = start_lines

    def get_entries(self, file_path: str) -> list[tuple[int, int, int, str]] | None:
        return self._entries.get(file_path)

    def get_start_lines(self, file_path: str) -> list[int] | None:
        return self._start_lines.get(file_path)


def build_file_symbol_index(
    graph: KnowledgeGraph,
    labels: tuple[NodeLabel, ...],
) -> FileSymbolIndex:
    """Build a per-file sorted interval index for fast containment lookups.

    For each file, symbols are stored as ``(start_line, end_line, span, node_id)``
    tuples sorted by ``start_line``.  A pre-computed ``start_lines`` list per file
    avoids redundant list creation during lookups.

    Args:
        graph: The knowledge graph containing parsed symbol nodes.
        labels: Which node labels to include in the index.

    Returns:
        A :class:`FileSymbolIndex` with entries and pre-computed start lines.
    """
    entries: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)

    for label in labels:
        for node in graph.get_nodes_by_label(label):
            if node.file_path and node.start_line > 0:
                span = node.end_line - node.start_line
                entries[node.file_path].append((node.start_line, node.end_line, span, node.id))

    for file_entries in entries.values():
        file_entries.sort(key=lambda t: t[0])

    start_lines: dict[str, list[int]] = {
        fp: [e[0] for e in file_entries] for fp, file_entries in entries.items()
    }

    return FileSymbolIndex(entries, start_lines)


def find_containing_symbol(
    line: int,
    file_path: str,
    file_symbol_index: FileSymbolIndex,
) -> str | None:
    """Find the most specific symbol whose line range contains *line*.

    Uses binary search on the pre-built index for O(log N) lookup instead
    of scanning all nodes.

    Args:
        line: The source line number to look up.
        file_path: Path to the file containing the line.
        file_symbol_index: Pre-built index from :func:`build_file_symbol_index`.

    Returns:
        The node ID of the most specific (smallest span) containing symbol,
        or ``None`` if no symbol contains the given line.
    """
    entries = file_symbol_index.get_entries(file_path)
    if not entries:
        return None

    # Binary search: find the rightmost entry whose start_line <= line.
    start_lines = file_symbol_index.get_start_lines(file_path)
    if not start_lines:
        return None
    idx = bisect.bisect_right(start_lines, line) - 1

    best_id: str | None = None
    best_span = float("inf")

    # Scan a small window around idx to handle nested/overlapping symbols.
    search_start = max(0, idx - 10)
    search_end = min(len(entries), idx + 5)

    for i in range(search_start, search_end):
        start, end, span, nid = entries[i]
        if start <= line <= end and span < best_span:
            best_span = span
            best_id = nid

    return best_id
