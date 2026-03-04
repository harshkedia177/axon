"""Tests for the import resolution phase (Phase 4)."""

from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.imports import (
    build_file_index,
    process_imports,
    resolve_import_path,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.parsers.base import ImportInfo, ParseResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FILE_PATHS = [
    # Python files
    ("src/auth/validate.py", "python"),
    ("src/auth/utils.py", "python"),
    ("src/auth/__init__.py", "python"),
    ("src/models/user.py", "python"),
    ("src/models/__init__.py", "python"),
    ("src/app.py", "python"),
    # TypeScript files
    ("lib/index.ts", "typescript"),
    ("lib/utils.ts", "typescript"),
    ("lib/models/user.ts", "typescript"),
    ("lib/models/index.ts", "typescript"),
]


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph pre-populated with File nodes for testing."""
    g = KnowledgeGraph()
    for path, language in _FILE_PATHS:
        node_id = generate_id(NodeLabel.FILE, path)
        g.add_node(
            GraphNode(
                id=node_id,
                label=NodeLabel.FILE,
                name=path.rsplit("/", 1)[-1],
                file_path=path,
                language=language,
            )
        )
    return g


@pytest.fixture()
def file_index(graph: KnowledgeGraph) -> dict[str, str]:
    """Return the file index built from the fixture graph."""
    return build_file_index(graph)


# ---------------------------------------------------------------------------
# build_file_index
# ---------------------------------------------------------------------------


class TestBuildFileIndex:
    """build_file_index creates correct mapping from graph File nodes."""

    def test_build_file_index(self, graph: KnowledgeGraph) -> None:
        index = build_file_index(graph)

        assert len(index) == len(_FILE_PATHS)
        for path, _ in _FILE_PATHS:
            assert path in index
            assert index[path] == generate_id(NodeLabel.FILE, path)

    def test_build_file_index_empty_graph(self) -> None:
        g = KnowledgeGraph()
        index = build_file_index(g)
        assert index == {}

    def test_build_file_index_ignores_non_file_nodes(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FOLDER, "src"),
                label=NodeLabel.FOLDER,
                name="src",
                file_path="src",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/app.py"),
                label=NodeLabel.FILE,
                name="app.py",
                file_path="src/app.py",
                language="python",
            )
        )
        index = build_file_index(g)
        assert len(index) == 1
        assert "src/app.py" in index


# ---------------------------------------------------------------------------
# resolve_import_path — Python
# ---------------------------------------------------------------------------


class TestResolvePythonRelativeImport:
    """from .utils import helper in src/auth/validate.py -> src/auth/utils.py."""

    def test_resolve_python_relative_import(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module=".utils", names=["helper"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "src/auth/utils.py")
        assert result == expected_id


class TestResolvePythonParentRelative:
    """from ..models import User in src/auth/validate.py -> src/models/__init__.py."""

    def test_resolve_python_parent_relative(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="..models", names=["User"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "src/models/__init__.py")
        assert result == expected_id

    def test_resolve_python_parent_relative_direct_module(self) -> None:
        """When models.py exists instead of models/__init__.py."""
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/auth/validate.py"),
                label=NodeLabel.FILE,
                name="validate.py",
                file_path="src/auth/validate.py",
                language="python",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/models.py"),
                label=NodeLabel.FILE,
                name="models.py",
                file_path="src/models.py",
                language="python",
            )
        )
        index = build_file_index(g)

        imp = ImportInfo(module="..models", names=["User"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, index)

        expected_id = generate_id(NodeLabel.FILE, "src/models.py")
        assert result == expected_id


class TestResolvePythonExternal:
    """import os or from os.path import join -> returns None (external)."""

    def test_resolve_python_external_import(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="os", names=[], is_relative=False)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)
        assert result is None

    def test_resolve_python_external_from_import(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="os.path", names=["join"], is_relative=False)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_import_path — TypeScript / JavaScript
# ---------------------------------------------------------------------------


class TestResolveTsRelative:
    """import { foo } from './utils' in lib/index.ts -> lib/utils.ts."""

    def test_resolve_ts_relative(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="./utils", names=["foo"], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "lib/utils.ts")
        assert result == expected_id


class TestResolveTsDirectoryIndex:
    """import { User } from './models' in lib/index.ts -> lib/models/index.ts."""

    def test_resolve_ts_directory_index(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="./models", names=["User"], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "lib/models/index.ts")
        assert result == expected_id


class TestResolveTsExternal:
    """import express from 'express' -> returns None (external)."""

    def test_resolve_ts_external(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="express", names=["express"], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)
        assert result is None

    def test_resolve_ts_scoped_external(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="@types/node", names=[], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)
        assert result is None


# ---------------------------------------------------------------------------
# process_imports — Integration
# ---------------------------------------------------------------------------


class TestProcessImportsCreatesRelationships:
    """process_imports creates IMPORTS edges in the graph."""

    def test_process_imports_creates_relationships(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1

        rel = imports_rels[0]
        assert rel.source == generate_id(NodeLabel.FILE, "src/auth/validate.py")
        assert rel.target == generate_id(NodeLabel.FILE, "src/auth/utils.py")
        assert rel.properties["symbols"] == "helper"

    def test_process_imports_relationship_id_format(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1
        assert imports_rels[0].id.startswith("imports:")
        assert "->" in imports_rels[0].id

    def test_process_imports_skips_external(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(module="os", names=["path"], is_relative=False),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 0

    def test_process_imports_multiple_files(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
            FileParseData(
                file_path="lib/index.ts",
                language="typescript",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module="./utils",
                            names=["foo"],
                            is_relative=False,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 2


class TestProcessImportsNoDuplicates:
    """Same import twice does not create duplicate edges."""

    def test_process_imports_no_duplicates(self, graph: KnowledgeGraph) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                        ImportInfo(
                            module=".utils",
                            names=["other_func"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1

    def test_process_imports_no_duplicates_across_parse_data(self, graph: KnowledgeGraph) -> None:
        """Duplicates are also prevented across separate FileParseData entries
        for the same file (e.g. if the same file appears twice)."""
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1
