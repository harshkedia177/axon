"""Tests for the type analysis phase (Phase 7)."""

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
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import build_name_index
from axon.core.ingestion.types import process_types
from axon.core.parsers.base import ParseResult, TypeRef

_TYPE_LABELS = (NodeLabel.CLASS, NodeLabel.INTERFACE, NodeLabel.TYPE_ALIAS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_file_node(graph: KnowledgeGraph, path: str) -> str:
    """Add a File node and return its ID."""
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
    """Add a symbol node with a DEFINES relationship from the file node."""
    symbol_name = f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the test fixture specification.

    File: src/auth.py
        Function: validate (lines 1-10)

    File: src/models.py
        Class: User (lines 1-20)
        Class: Config (lines 22-40)

    File: src/types.ts
        Interface: AuthResult (lines 1-10)
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/models.py")
    _add_file_node(g, "src/types.ts")

    # Symbols in src/auth.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10)

    # Symbols in src/models.py
    _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "User", 1, 20)
    _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "Config", 22, 40)

    # Symbols in src/types.ts
    _add_symbol_node(g, NodeLabel.INTERFACE, "src/types.ts", "AuthResult", 1, 10)

    return g


@pytest.fixture()
def parse_data() -> list[FileParseData]:
    """Parse data with type refs matching the fixture specification.

    src/auth.py: User param at line 2, Config param at line 2.
    """
    return [
        FileParseData(
            file_path="src/auth.py",
            language="python",
            parse_result=ParseResult(
                type_refs=[
                    TypeRef(name="User", kind="param", line=2, param_name="user"),
                    TypeRef(name="Config", kind="param", line=2, param_name="config"),
                ],
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# build_type_index
# ---------------------------------------------------------------------------


class TestBuildTypeIndex:
    """build_type_index creates correct mapping from graph type nodes."""

    def test_build_type_index(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _TYPE_LABELS)

        # Class and Interface nodes should appear.
        assert "User" in index
        assert "Config" in index
        assert "AuthResult" in index

        # Each name maps to exactly one node ID.
        assert len(index["User"]) == 1
        assert len(index["Config"]) == 1
        assert len(index["AuthResult"]) == 1

        # IDs match expected generate_id output.
        expected_user = generate_id(NodeLabel.CLASS, "src/models.py", "User")
        assert index["User"] == [expected_user]

        expected_auth_result = generate_id(NodeLabel.INTERFACE, "src/types.ts", "AuthResult")
        assert index["AuthResult"] == [expected_auth_result]

        # Function nodes should NOT appear in the type index.
        assert "validate" not in index

    def test_build_type_index_includes_type_alias(self) -> None:
        """TypeAlias nodes are included."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/aliases.py")
        _add_symbol_node(g, NodeLabel.TYPE_ALIAS, "src/aliases.py", "UserID", 1, 1)

        index = build_name_index(g, _TYPE_LABELS)
        assert "UserID" in index
        assert len(index["UserID"]) == 1

    def test_build_type_index_multiple_same_name(self) -> None:
        """Multiple types with the same name produce a list with all IDs."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/a.py")
        _add_file_node(g, "src/b.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/a.py", "Base", 1, 10)
        _add_symbol_node(g, NodeLabel.CLASS, "src/b.py", "Base", 1, 10)

        index = build_name_index(g, _TYPE_LABELS)
        assert "Base" in index
        assert len(index["Base"]) == 2


# ---------------------------------------------------------------------------
# process_types — creates USES_TYPE relationships
# ---------------------------------------------------------------------------


class TestProcessTypesCreatesUsesType:
    """process_types creates USES_TYPE edges in the graph."""

    def test_process_types_creates_uses_type(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_types(parse_data, graph)

        uses_rels = graph.get_relationships_by_type(RelType.USES_TYPE)
        assert len(uses_rels) == 2

        # Collect source->target pairs.
        pairs = {(r.source, r.target) for r in uses_rels}

        validate_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "validate")
        user_id = generate_id(NodeLabel.CLASS, "src/models.py", "User")
        config_id = generate_id(NodeLabel.CLASS, "src/models.py", "Config")

        assert (validate_id, user_id) in pairs
        assert (validate_id, config_id) in pairs


# ---------------------------------------------------------------------------
# process_types — role property
# ---------------------------------------------------------------------------


class TestProcessTypesRoleProperty:
    """Role property is set correctly on USES_TYPE relationships."""

    def test_process_types_role_property(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_types(parse_data, graph)

        uses_rels = graph.get_relationships_by_type(RelType.USES_TYPE)

        # Both references in the fixture are "param" kind.
        for rel in uses_rels:
            assert rel.properties["role"] == "param"


# ---------------------------------------------------------------------------
# process_types — unresolved types are skipped
# ---------------------------------------------------------------------------


class TestProcessTypesUnresolvedSkipped:
    """Unknown type names don't crash and produce no relationships."""

    def test_process_types_unresolved_skipped(self, graph: KnowledgeGraph) -> None:
        unresolved_data = [
            FileParseData(
                file_path="src/auth.py",
                language="python",
                parse_result=ParseResult(
                    type_refs=[
                        TypeRef(
                            name="NonExistentType",
                            kind="param",
                            line=2,
                            param_name="x",
                        ),
                    ],
                ),
            ),
        ]

        process_types(unresolved_data, graph)

        uses_rels = graph.get_relationships_by_type(RelType.USES_TYPE)
        assert len(uses_rels) == 0


# ---------------------------------------------------------------------------
# process_types — no duplicates
# ---------------------------------------------------------------------------


class TestProcessTypesNoDuplicates:
    """Same type used twice in the same role doesn't duplicate edges."""

    def test_process_types_no_duplicates(self, graph: KnowledgeGraph) -> None:
        # Two param references to User inside validate (same role).
        duplicate_data = [
            FileParseData(
                file_path="src/auth.py",
                language="python",
                parse_result=ParseResult(
                    type_refs=[
                        TypeRef(name="User", kind="param", line=2, param_name="user"),
                        TypeRef(name="User", kind="param", line=3, param_name="other_user"),
                    ],
                ),
            ),
        ]

        process_types(duplicate_data, graph)

        uses_rels = graph.get_relationships_by_type(RelType.USES_TYPE)
        # Both refs resolve to validate -> User with role "param", but only
        # one relationship should exist due to the ID-based dedup.
        assert len(uses_rels) == 1


# ---------------------------------------------------------------------------
# process_types — return type creates relationship
# ---------------------------------------------------------------------------


class TestProcessTypesReturnType:
    """Return type annotation creates USES_TYPE with role='return'."""

    def test_process_types_return_type(self, graph: KnowledgeGraph) -> None:
        return_data = [
            FileParseData(
                file_path="src/auth.py",
                language="python",
                parse_result=ParseResult(
                    type_refs=[
                        TypeRef(name="User", kind="return", line=1),
                    ],
                ),
            ),
        ]

        process_types(return_data, graph)

        uses_rels = graph.get_relationships_by_type(RelType.USES_TYPE)
        assert len(uses_rels) == 1

        rel = uses_rels[0]
        validate_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "validate")
        user_id = generate_id(NodeLabel.CLASS, "src/models.py", "User")

        assert rel.source == validate_id
        assert rel.target == user_id
        assert rel.properties["role"] == "return"
