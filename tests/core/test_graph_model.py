"""Tests for the knowledge graph data model."""

from __future__ import annotations

import pytest

from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)

# ---------------------------------------------------------------------------
# NodeLabel enum
# ---------------------------------------------------------------------------


class TestNodeLabel:
    """Verify every expected node label exists and has correct string values."""

    EXPECTED = [
        "FILE",
        "FOLDER",
        "FUNCTION",
        "CLASS",
        "METHOD",
        "INTERFACE",
        "TYPE_ALIAS",
        "ENUM",
        "COMMUNITY",
        "PROCESS",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_label_exists(self, name: str) -> None:
        assert hasattr(NodeLabel, name), f"NodeLabel.{name} is missing"

    def test_label_count(self) -> None:
        assert len(NodeLabel) == len(self.EXPECTED)

    def test_label_values_are_lowercase(self) -> None:
        for label in NodeLabel:
            assert label.value == label.name.lower()


# ---------------------------------------------------------------------------
# RelType enum
# ---------------------------------------------------------------------------


class TestRelType:
    """Verify every expected relationship type exists."""

    EXPECTED = [
        "CONTAINS",
        "DEFINES",
        "CALLS",
        "IMPORTS",
        "EXTENDS",
        "IMPLEMENTS",
        "MEMBER_OF",
        "STEP_IN_PROCESS",
        "USES_TYPE",
        "EXPORTS",
        "COUPLED_WITH",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_rel_type_exists(self, name: str) -> None:
        assert hasattr(RelType, name), f"RelType.{name} is missing"

    def test_rel_type_count(self) -> None:
        assert len(RelType) == len(self.EXPECTED)

    def test_rel_type_values_are_lowercase(self) -> None:
        for rel in RelType:
            assert rel.value == rel.name.lower()


# ---------------------------------------------------------------------------
# generate_id
# ---------------------------------------------------------------------------


class TestGenerateId:
    """Verify deterministic ID generation."""

    def test_with_symbol_name(self) -> None:
        result = generate_id(NodeLabel.FUNCTION, "src/main.py", "do_stuff")
        assert result == "function:src/main.py:do_stuff"

    def test_without_symbol_name(self) -> None:
        result = generate_id(NodeLabel.FILE, "src/main.py")
        assert result == "file:src/main.py:"

    def test_explicit_empty_symbol(self) -> None:
        result = generate_id(NodeLabel.FOLDER, "src/", "")
        assert result == "folder:src/:"

    def test_uses_label_value(self) -> None:
        result = generate_id(NodeLabel.TYPE_ALIAS, "types.py", "MyType")
        assert result == "type_alias:types.py:MyType"


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------


class TestGraphNode:
    """Verify GraphNode dataclass defaults and construction."""

    def test_minimal_creation(self) -> None:
        node = GraphNode(id="test:id", label=NodeLabel.FILE, name="main.py")
        assert node.id == "test:id"
        assert node.label is NodeLabel.FILE
        assert node.name == "main.py"

    def test_defaults(self) -> None:
        node = GraphNode(id="n1", label=NodeLabel.CLASS, name="Foo")
        assert node.file_path == ""
        assert node.start_line == 0
        assert node.end_line == 0
        assert node.content == ""
        assert node.signature == ""
        assert node.language == ""
        assert node.class_name == ""
        assert node.is_dead is False
        assert node.is_entry_point is False
        assert node.is_exported is False
        assert node.properties == {}

    def test_properties_default_is_independent(self) -> None:
        """Mutable default must not be shared across instances."""
        a = GraphNode(id="a", label=NodeLabel.FILE, name="a.py")
        b = GraphNode(id="b", label=NodeLabel.FILE, name="b.py")
        a.properties["key"] = "val"
        assert "key" not in b.properties

    def test_full_creation(self) -> None:
        node = GraphNode(
            id="function:app.py:main",
            label=NodeLabel.FUNCTION,
            name="main",
            file_path="app.py",
            start_line=10,
            end_line=25,
            content="def main(): ...",
            signature="def main() -> None",
            language="python",
            class_name="",
            is_dead=False,
            is_entry_point=True,
            is_exported=True,
            properties={"complexity": 3},
        )
        assert node.start_line == 10
        assert node.end_line == 25
        assert node.is_entry_point is True
        assert node.properties["complexity"] == 3


# ---------------------------------------------------------------------------
# GraphRelationship
# ---------------------------------------------------------------------------


class TestGraphRelationship:
    """Verify GraphRelationship dataclass."""

    def test_creation(self) -> None:
        rel = GraphRelationship(
            id="r1",
            type=RelType.CALLS,
            source="function:a.py:foo",
            target="function:b.py:bar",
        )
        assert rel.id == "r1"
        assert rel.type is RelType.CALLS
        assert rel.source == "function:a.py:foo"
        assert rel.target == "function:b.py:bar"

    def test_default_properties(self) -> None:
        rel = GraphRelationship(
            id="r2",
            type=RelType.IMPORTS,
            source="file:a.py:",
            target="file:b.py:",
        )
        assert rel.properties == {}

    def test_properties_default_is_independent(self) -> None:
        a = GraphRelationship(id="a", type=RelType.CALLS, source="s", target="t")
        b = GraphRelationship(id="b", type=RelType.CALLS, source="s", target="t")
        a.properties["weight"] = 5
        assert "weight" not in b.properties

    def test_with_properties(self) -> None:
        rel = GraphRelationship(
            id="r3",
            type=RelType.COUPLED_WITH,
            source="file:x.py:",
            target="file:y.py:",
            properties={"score": 0.85},
        )
        assert rel.properties["score"] == 0.85
