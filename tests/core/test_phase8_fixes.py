"""Tests for Phase 8: dynamic dispatch, example files, same-file blocklist override.

Three false-positive categories addressed:
1. PHP string-in-array method callbacks ($this->$method() from property strings)
2. Example/template file exemption (config.example.php)
3. Same-file blocklist override (user-defined function named 'close')
"""

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
from axon.core.ingestion.calls import process_calls
from axon.core.ingestion.dead_code import (
    _is_example_file,
    _is_exempt,
    process_dead_code,
)
from axon.core.ingestion.parser_phase import FileParseData, parse_file
from axon.core.parsers.base import CallInfo, ParseResult, SymbolInfo
from axon.core.parsers.php import PhpParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_file_node(graph: KnowledgeGraph, path: str) -> str:
    node_id = generate_id(NodeLabel.FILE, path)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=NodeLabel.FILE,
            name=path.rsplit("/", 1)[-1],
            file_path=path,
        )
    )
    return node_id


def _add_symbol_node(
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
    file_id = generate_id(NodeLabel.FILE, file_path)
    graph.add_relationship(
        GraphRelationship(
            id=f"defines:{file_id}->{node_id}",
            type=RelType.DEFINES,
            source=file_id,
            target=node_id,
        )
    )
    return node_id


# ---------------------------------------------------------------------------
# Fix 1: PHP string-in-array method callbacks
# ---------------------------------------------------------------------------


class TestPhpStringMethodRefs:
    """String values in class properties that match method names emit calls."""

    def test_string_callback_in_property_array(self) -> None:
        """'fixSqlInjection' in property array emits a synthetic call."""
        parser = PhpParser()
        code = """<?php
class AutoFix {
    private $fixPatterns = [
        'sql_injection' => [
            'replace_callback' => 'fixSqlInjection'
        ]
    ];

    public function apply() {
        $method = $fix['replace_callback'];
        $this->$method($code, $fix['match']);
    }

    private function fixSqlInjection($code, $match) {
        return $code;
    }
}
"""
        result = parser.parse(code, "auto-fix.php")
        call_names = [(c.name, c.receiver) for c in result.calls]
        assert ("fixSqlInjection", "this") in call_names

    def test_non_method_strings_ignored(self) -> None:
        """Strings that don't match method names are not emitted as calls."""
        parser = PhpParser()
        code = """<?php
class Fixer {
    private $config = [
        'description' => 'Fix SQL injection',
        'severity' => 'critical'
    ];

    public function fix() { return true; }
}
"""
        result = parser.parse(code, "test.php")
        call_names = [c.name for c in result.calls]
        assert "Fix SQL injection" not in call_names
        assert "critical" not in call_names

    def test_dunder_strings_not_emitted(self) -> None:
        """Strings starting with __ are excluded (magic methods handled elsewhere)."""
        parser = PhpParser()
        code = """<?php
class Foo {
    private $hooks = ['__construct', '__toString'];

    public function __construct() {}
    public function __toString() { return ''; }
}
"""
        result = parser.parse(code, "test.php")
        # __construct and __toString should NOT be emitted as synthetic calls
        # from the string refs (they're already handled as magic methods).
        synthetic_calls = [
            c for c in result.calls
            if c.name.startswith("__") and c.receiver == "this"
        ]
        assert len(synthetic_calls) == 0

    def test_string_callback_creates_calls_edge(self) -> None:
        """End-to-end: string callback resolves to CALLS edge via process_calls."""
        g = KnowledgeGraph()

        _add_file_node(g, "api/fixer.php")
        _add_symbol_node(
            g, NodeLabel.CLASS, "api/fixer.php", "Fixer", 1, 20,
        )
        _add_symbol_node(
            g, NodeLabel.METHOD, "api/fixer.php", "apply", 3, 10,
            class_name="Fixer",
        )
        _add_symbol_node(
            g, NodeLabel.METHOD, "api/fixer.php", "fixBug", 12, 18,
            class_name="Fixer",
        )

        parse_data = [
            FileParseData(
                file_path="api/fixer.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="Fixer", kind="class",
                            start_line=1, end_line=20, content="",
                        ),
                        SymbolInfo(
                            name="apply", kind="method",
                            start_line=3, end_line=10, content="",
                            class_name="Fixer",
                        ),
                        SymbolInfo(
                            name="fixBug", kind="method",
                            start_line=12, end_line=18, content="",
                            class_name="Fixer",
                        ),
                    ],
                    calls=[
                        # Synthetic call from string ref
                        CallInfo(name="fixBug", line=5, receiver="this"),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}
        expected_id = generate_id(
            NodeLabel.METHOD, "api/fixer.php", "Fixer.fixBug",
        )
        assert expected_id in targets


# ---------------------------------------------------------------------------
# Fix 2: Example/template file exemption
# ---------------------------------------------------------------------------


class TestExampleFileExemption:
    """Symbols in example/template files are exempt from dead-code flagging."""

    def test_is_example_file_matches(self) -> None:
        assert _is_example_file("api/config/config.example.php")
        assert _is_example_file("settings.sample.py")
        assert _is_example_file("env.template.yaml")
        assert _is_example_file("docs/examples/auth.py")
        assert _is_example_file("src/example/demo.ts")

    def test_is_example_file_rejects(self) -> None:
        assert not _is_example_file("api/config/config.php")
        assert not _is_example_file("src/app.py")
        assert not _is_example_file("api/example_handler.php")

    def test_example_file_exempt_from_dead_code(self) -> None:
        assert _is_exempt(
            "getPDO", is_entry_point=False, is_exported=False,
            file_path="api/config/config.example.php",
        )

    def test_normal_file_not_exempt(self) -> None:
        assert not _is_exempt(
            "getPDO", is_entry_point=False, is_exported=False,
            file_path="api/config/config.php",
        )

    def test_example_file_symbols_not_flagged_dead(self) -> None:
        """Symbols in config.example.php are never flagged as dead."""
        g = KnowledgeGraph()

        _add_file_node(g, "config.example.php")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "config.example.php", "getPDO", 1, 10,
        )

        dead_count = process_dead_code(g)
        node = g.get_node(node_id)
        assert node is not None
        assert not node.is_dead
        assert dead_count == 0


# ---------------------------------------------------------------------------
# Fix 3: Same-file blocklist override
# ---------------------------------------------------------------------------


class TestSameFileBlocklistOverride:
    """Blocklisted names still resolve when target is in the same file."""

    def test_same_file_close_creates_edge(self) -> None:
        """A function named 'close' called in the same file creates a CALLS edge."""
        g = KnowledgeGraph()

        _add_file_node(g, "sidebar.js")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "sidebar.js", "showModal", 1, 20,
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "sidebar.js", "close", 5, 8,
        )

        parse_data = [
            FileParseData(
                file_path="sidebar.js",
                language="javascript",
                parse_result=ParseResult(
                    calls=[
                        # direct call: close()
                        CallInfo(name="close", line=15),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}
        expected_id = generate_id(
            NodeLabel.FUNCTION, "sidebar.js", "close",
        )
        assert expected_id in targets

    def test_blocklisted_callback_same_file_creates_edge(self) -> None:
        """A blocklisted name passed as callback creates edge if same-file."""
        g = KnowledgeGraph()

        _add_file_node(g, "sidebar.js")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "sidebar.js", "showModal", 1, 20,
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "sidebar.js", "close", 5, 8,
        )

        parse_data = [
            FileParseData(
                file_path="sidebar.js",
                language="javascript",
                parse_result=ParseResult(
                    calls=[
                        # addEventListener('click', close) — close as callback
                        CallInfo(
                            name="addEventListener", line=15,
                            receiver="btn", arguments=["close"],
                        ),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}
        expected_id = generate_id(
            NodeLabel.FUNCTION, "sidebar.js", "close",
        )
        assert expected_id in targets

    def test_blocklisted_cross_file_still_blocked(self) -> None:
        """A blocklisted name in a DIFFERENT file does NOT create an edge."""
        g = KnowledgeGraph()

        _add_file_node(g, "a.js")
        _add_file_node(g, "b.js")
        _add_symbol_node(g, NodeLabel.FUNCTION, "a.js", "caller", 1, 10)
        _add_symbol_node(g, NodeLabel.FUNCTION, "b.js", "close", 1, 5)

        parse_data = [
            FileParseData(
                file_path="a.js",
                language="javascript",
                parse_result=ParseResult(
                    calls=[CallInfo(name="close", line=5)],
                ),
            ),
        ]

        process_calls(parse_data, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 0, "Blocklisted cross-file call should not create edge"
