"""Symbol lookup unit tests.

Covers build_file_symbol_index, find_containing_symbol edge cases
(binary search boundaries, nested symbols, empty files, line before/after
all symbols), and FileSymbolIndex accessors.
"""

from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, NodeLabel, generate_id
from axon.core.ingestion.symbol_lookup import (
    FileSymbolIndex,
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)

_CALLABLE_LABELS = (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_symbol(
    graph: KnowledgeGraph,
    label: NodeLabel,
    file_path: str,
    name: str,
    start_line: int,
    end_line: int,
    class_name: str = "",
) -> str:
    symbol_name = (
        f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
    )
    node_id = generate_id(label, file_path, symbol_name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=label,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            class_name=class_name,
        )
    )
    return node_id


# ---------------------------------------------------------------------------
# build_file_symbol_index
# ---------------------------------------------------------------------------


class TestBuildFileSymbolIndex:
    """build_file_symbol_index creates a correct interval index."""

    def test_basic_index(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "bar", 12, 20)

        idx = build_file_symbol_index(g, _CALLABLE_LABELS)
        entries = idx.get_entries("src/a.py")
        assert entries is not None
        assert len(entries) == 2

    def test_entries_sorted_by_start_line(self) -> None:
        g = KnowledgeGraph()
        # Add in reverse order.
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "bar", 20, 30)
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)

        idx = build_file_symbol_index(g, _CALLABLE_LABELS)
        entries = idx.get_entries("src/a.py")
        assert entries is not None
        assert entries[0][0] == 1   # foo starts at line 1
        assert entries[1][0] == 20  # bar starts at line 20

    def test_start_lines_precomputed(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "bar", 12, 20)

        idx = build_file_symbol_index(g, _CALLABLE_LABELS)
        start_lines = idx.get_start_lines("src/a.py")
        assert start_lines == [1, 12]

    def test_empty_graph(self) -> None:
        g = KnowledgeGraph()
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)
        assert idx.get_entries("src/a.py") is None
        assert idx.get_start_lines("src/a.py") is None

    def test_multiple_files(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        _add_symbol(g, NodeLabel.FUNCTION, "src/b.py", "bar", 1, 5)

        idx = build_file_symbol_index(g, _CALLABLE_LABELS)
        assert idx.get_entries("src/a.py") is not None
        assert idx.get_entries("src/b.py") is not None
        assert len(idx.get_entries("src/a.py")) == 1
        assert len(idx.get_entries("src/b.py")) == 1

    def test_symbols_with_zero_start_line_excluded(self) -> None:
        g = KnowledgeGraph()
        # start_line=0 means no line info — should be excluded.
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "nolines", 0, 0)
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "haslines", 5, 10)

        idx = build_file_symbol_index(g, _CALLABLE_LABELS)
        entries = idx.get_entries("src/a.py")
        assert entries is not None
        assert len(entries) == 1
        assert entries[0][3].endswith(":haslines")

    def test_span_computed_correctly(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 5, 15)

        idx = build_file_symbol_index(g, _CALLABLE_LABELS)
        entries = idx.get_entries("src/a.py")
        # span = end_line - start_line = 15 - 5 = 10
        assert entries[0][2] == 10


# ---------------------------------------------------------------------------
# find_containing_symbol — basic cases
# ---------------------------------------------------------------------------


class TestFindContainingSymbolBasic:
    """Basic containment lookups."""

    def test_line_inside_symbol(self) -> None:
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(5, "src/a.py", idx)
        assert result == func_id

    def test_line_on_start_boundary(self) -> None:
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 10, 20)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(10, "src/a.py", idx)
        assert result == func_id

    def test_line_on_end_boundary(self) -> None:
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 10, 20)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(20, "src/a.py", idx)
        assert result == func_id

    def test_line_between_two_symbols(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "bar", 15, 25)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        # Line 12 is between foo (1-10) and bar (15-25).
        result = find_containing_symbol(12, "src/a.py", idx)
        assert result is None

    def test_correct_symbol_selected(self) -> None:
        g = KnowledgeGraph()
        foo_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        bar_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "bar", 15, 25)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        assert find_containing_symbol(5, "src/a.py", idx) == foo_id
        assert find_containing_symbol(20, "src/a.py", idx) == bar_id


# ---------------------------------------------------------------------------
# find_containing_symbol — edge cases
# ---------------------------------------------------------------------------


class TestFindContainingSymbolEdgeCases:
    """Edge cases: line before all, after all, unknown file, nested."""

    def test_line_before_all_symbols(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 10, 20)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(1, "src/a.py", idx)
        assert result is None

    def test_line_after_all_symbols(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(50, "src/a.py", idx)
        assert result is None

    def test_unknown_file(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 10)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(5, "src/unknown.py", idx)
        assert result is None

    def test_empty_file(self) -> None:
        g = KnowledgeGraph()
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(5, "src/a.py", idx)
        assert result is None

    def test_nested_symbol_selects_smallest_span(self) -> None:
        """Inner function (smaller span) is preferred over outer."""
        g = KnowledgeGraph()
        outer_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "outer", 1, 50)
        inner_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "inner", 10, 15)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        # Line 12 is inside both outer (1-50) and inner (10-15).
        # Inner has smaller span (5 vs 49) so it should win.
        result = find_containing_symbol(12, "src/a.py", idx)
        assert result == inner_id

    def test_line_in_outer_but_not_inner(self) -> None:
        """Line outside inner but inside outer selects outer."""
        g = KnowledgeGraph()
        outer_id = _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "outer", 1, 50)
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "inner", 10, 15)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        # Line 30 is inside outer (1-50) but not inner (10-15).
        result = find_containing_symbol(30, "src/a.py", idx)
        assert result == outer_id

    def test_class_with_method_selects_method(self) -> None:
        """Method inside a class is more specific than the class."""
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.CLASS, "src/a.py", "MyClass", 1, 50)
        method_id = _add_symbol(
            g, NodeLabel.METHOD, "src/a.py", "do_work", 5, 15,
            class_name="MyClass",
        )
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        result = find_containing_symbol(10, "src/a.py", idx)
        assert result == method_id

    def test_many_small_symbols(self) -> None:
        """Works with more than 10 adjacent symbols (tests the ±10 window)."""
        g = KnowledgeGraph()
        ids = []
        for i in range(20):
            start = i * 5 + 1
            end = start + 3
            sym_id = _add_symbol(
                g, NodeLabel.FUNCTION, "src/big.py", f"func_{i}", start, end
            )
            ids.append(sym_id)
        idx = build_file_symbol_index(g, _CALLABLE_LABELS)

        # Check the last symbol (index 19, starts at 96, ends at 99).
        result = find_containing_symbol(97, "src/big.py", idx)
        assert result == ids[19]

        # Check a symbol in the middle (index 10, starts at 51, ends at 54).
        result = find_containing_symbol(52, "src/big.py", idx)
        assert result == ids[10]


# ---------------------------------------------------------------------------
# build_name_index
# ---------------------------------------------------------------------------


class TestBuildNameIndex:
    """build_name_index basic functionality."""

    def test_groups_by_name(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "init", 1, 5)
        _add_symbol(g, NodeLabel.FUNCTION, "src/b.py", "init", 1, 5)

        idx = build_name_index(g, _CALLABLE_LABELS)
        assert "init" in idx
        assert len(idx["init"]) == 2

    def test_respects_label_filter(self) -> None:
        g = KnowledgeGraph()
        _add_symbol(g, NodeLabel.FUNCTION, "src/a.py", "foo", 1, 5)
        _add_symbol(g, NodeLabel.CLASS, "src/a.py", "Foo", 1, 20)

        # Only FUNCTION labels.
        idx = build_name_index(g, (NodeLabel.FUNCTION,))
        assert "foo" in idx
        assert "Foo" not in idx
