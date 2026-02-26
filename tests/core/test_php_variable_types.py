"""Tests for PHP factory return type inference (Phase 7).

Verifies that $var = factory() assignments are tracked and that
$var->method() calls resolve to methods on the factory's return types.
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
from axon.core.ingestion.parser_phase import FileParseData, parse_file
from axon.core.parsers.base import CallInfo, ParseResult, SymbolInfo, TypeRef
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
# PHP Parser: assignment tracking
# ---------------------------------------------------------------------------

class TestPhpAssignmentExtraction:
    """PhpParser extracts $var = function() assignments."""

    def test_function_call_assignment(self) -> None:
        """$client = getMessagingClient() is tracked."""
        parser = PhpParser()
        result = parser.parse(
            '<?php\n$client = getMessagingClient();\n',
            "test.php",
        )
        # Should have a __call__ sentinel since getMessagingClient is not
        # defined in the same file.
        assert "client" in result.variable_types
        assert result.variable_types["client"] == ["__call__getMessagingClient"]

    def test_new_expression_assignment(self) -> None:
        """$client = new LinqClient() resolves directly to the class."""
        parser = PhpParser()
        result = parser.parse(
            '<?php\n$client = new LinqClient();\n',
            "test.php",
        )
        assert "client" in result.variable_types
        assert result.variable_types["client"] == ["LinqClient"]

    def test_same_file_factory_resolves_return_types(self) -> None:
        """Factory defined in the same file resolves to return types."""
        parser = PhpParser()
        code = """<?php
function getClient(): LinqClient|BlueBubblesClient {
    return new LinqClient();
}
$client = getClient();
"""
        result = parser.parse(code, "test.php")
        assert "client" in result.variable_types
        types = result.variable_types["client"]
        assert "LinqClient" in types
        assert "BlueBubblesClient" in types

    def test_no_assignment_no_variable_types(self) -> None:
        """Files without assignments produce empty variable_types."""
        parser = PhpParser()
        result = parser.parse(
            '<?php\necho "hello";\n',
            "test.php",
        )
        assert result.variable_types == {}

    def test_non_variable_lhs_ignored(self) -> None:
        """$obj->prop = func() is not tracked (only simple $var)."""
        parser = PhpParser()
        result = parser.parse(
            '<?php\n$obj->prop = getClient();\n',
            "test.php",
        )
        assert "prop" not in result.variable_types


# ---------------------------------------------------------------------------
# PHP Parser: union return type extraction
# ---------------------------------------------------------------------------

class TestPhpUnionReturnType:
    """PhpParser extracts union return types."""

    def test_union_return_type(self) -> None:
        """function foo(): A|B extracts both type refs."""
        parser = PhpParser()
        code = """<?php
function foo(): TypeA|TypeB {
    return new TypeA();
}
"""
        result = parser.parse(code, "test.php")
        return_types = [t.name for t in result.type_refs if t.kind == "return"]
        assert "TypeA" in return_types
        assert "TypeB" in return_types

    def test_single_return_type_still_works(self) -> None:
        """function foo(): TypeA extracts one type ref (regression)."""
        parser = PhpParser()
        code = """<?php
function foo(): SomeClass {
    return new SomeClass();
}
"""
        result = parser.parse(code, "test.php")
        return_types = [t.name for t in result.type_refs if t.kind == "return"]
        assert "SomeClass" in return_types


# ---------------------------------------------------------------------------
# Cross-file factory type inference via process_calls
# ---------------------------------------------------------------------------

class TestCrossFileFactoryResolution:
    """$client = getFactory() resolves across files via process_calls."""

    def test_cross_file_factory_creates_calls_edge(self) -> None:
        """$client->sendText() creates a CALLS edge to LinqClient.sendText."""
        g = KnowledgeGraph()

        # File 1: defines the factory with return type
        _add_file_node(g, "api/MessagingProvider.php")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "api/MessagingProvider.php",
            "getMessagingClient", 1, 5,
        )

        # File 2: defines LinqClient.sendTextMessage
        _add_file_node(g, "api/LinqClient.php")
        _add_symbol_node(
            g, NodeLabel.CLASS, "api/LinqClient.php", "LinqClient", 1, 50,
        )
        _add_symbol_node(
            g, NodeLabel.METHOD, "api/LinqClient.php", "sendTextMessage",
            10, 20, class_name="LinqClient",
        )

        # File 3: defines BlueBubblesClient.sendTextMessage
        _add_file_node(g, "api/BlueBubblesClient.php")
        _add_symbol_node(
            g, NodeLabel.CLASS, "api/BlueBubblesClient.php",
            "BlueBubblesClient", 1, 50,
        )
        _add_symbol_node(
            g, NodeLabel.METHOD, "api/BlueBubblesClient.php",
            "sendTextMessage", 10, 20, class_name="BlueBubblesClient",
        )

        # File 4: calls $client = getMessagingClient(); $client->sendTextMessage()
        _add_file_node(g, "api/messaging.php")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "api/messaging.php", "handleSend", 1, 20,
        )

        parse_data = [
            # MessagingProvider.php: factory with union return type
            FileParseData(
                file_path="api/MessagingProvider.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="getMessagingClient", kind="function",
                            start_line=1, end_line=5, content="",
                        ),
                    ],
                    type_refs=[
                        TypeRef(name="LinqClient", kind="return", line=1),
                        TypeRef(name="BlueBubblesClient", kind="return", line=1),
                    ],
                ),
            ),
            # messaging.php: calls the factory and uses the result
            FileParseData(
                file_path="api/messaging.php",
                language="php",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="getMessagingClient", line=5),
                        CallInfo(
                            name="sendTextMessage", line=8,
                            receiver="client",
                        ),
                    ],
                    variable_types={
                        "client": ["__call__getMessagingClient"],
                    },
                ),
            ),
            # LinqClient.php and BlueBubblesClient.php: no calls
            FileParseData(
                file_path="api/LinqClient.php",
                language="php",
                parse_result=ParseResult(),
            ),
            FileParseData(
                file_path="api/BlueBubblesClient.php",
                language="php",
                parse_result=ParseResult(),
            ),
        ]

        process_calls(parse_data, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}

        linq_method_id = generate_id(
            NodeLabel.METHOD, "api/LinqClient.php",
            "LinqClient.sendTextMessage",
        )
        bb_method_id = generate_id(
            NodeLabel.METHOD, "api/BlueBubblesClient.php",
            "BlueBubblesClient.sendTextMessage",
        )

        assert linq_method_id in targets, "Should create CALLS edge to LinqClient.sendTextMessage"
        assert bb_method_id in targets, "Should create CALLS edge to BlueBubblesClient.sendTextMessage"

    def test_new_expression_creates_calls_edge(self) -> None:
        """$client = new LinqClient(); $client->send() creates CALLS edge."""
        g = KnowledgeGraph()

        _add_file_node(g, "api/LinqClient.php")
        _add_symbol_node(
            g, NodeLabel.CLASS, "api/LinqClient.php", "LinqClient", 1, 50,
        )
        _add_symbol_node(
            g, NodeLabel.METHOD, "api/LinqClient.php", "send",
            10, 20, class_name="LinqClient",
        )

        _add_file_node(g, "api/usage.php")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "api/usage.php", "doSend", 1, 10,
        )

        parse_data = [
            FileParseData(
                file_path="api/usage.php",
                language="php",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="send", line=5, receiver="client"),
                    ],
                    variable_types={"client": ["LinqClient"]},
                ),
            ),
            FileParseData(
                file_path="api/LinqClient.php",
                language="php",
                parse_result=ParseResult(),
            ),
        ]

        process_calls(parse_data, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}

        expected_id = generate_id(
            NodeLabel.METHOD, "api/LinqClient.php", "LinqClient.send",
        )
        assert expected_id in targets


# ---------------------------------------------------------------------------
# End-to-end: PHP parse + call resolution
# ---------------------------------------------------------------------------

class TestPhpEndToEndFactoryInference:
    """Full pipeline: parse PHP → process calls → verify edges."""

    def test_same_file_factory_e2e(self) -> None:
        """Factory and usage in the same file resolves correctly."""
        g = KnowledgeGraph()

        code = """<?php
class MyClient {
    public function doWork() { return true; }
}

function getClient(): MyClient {
    return new MyClient();
}

function main() {
    $c = getClient();
    $c->doWork();
}
"""
        # Parse the file
        fpd = parse_file("test.php", code, "php")

        # Verify variable_types was populated
        assert "c" in fpd.parse_result.variable_types
        assert "MyClient" in fpd.parse_result.variable_types["c"]

        # Set up graph with the parsed symbols
        _add_file_node(g, "test.php")
        for sym in fpd.parse_result.symbols:
            label = {
                "function": NodeLabel.FUNCTION,
                "class": NodeLabel.CLASS,
                "method": NodeLabel.METHOD,
            }[sym.kind]
            _add_symbol_node(
                g, label, "test.php", sym.name,
                sym.start_line, sym.end_line, class_name=sym.class_name,
            )

        process_calls([fpd], g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}

        expected_id = generate_id(
            NodeLabel.METHOD, "test.php", "MyClient.doWork",
        )
        assert expected_id in targets, (
            f"Expected CALLS edge to MyClient.doWork. "
            f"Got targets: {targets}"
        )
