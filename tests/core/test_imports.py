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

    def test_resolve_python_relative_import(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module=".utils", names=["helper"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "src/auth/utils.py")
        assert result == expected_id


class TestResolvePythonParentRelative:
    """from ..models import User in src/auth/validate.py -> src/models/__init__.py."""

    def test_resolve_python_parent_relative(
        self, file_index: dict[str, str]
    ) -> None:
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

    def test_resolve_python_external_import(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="os", names=[], is_relative=False)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)
        assert result is None

    def test_resolve_python_external_from_import(
        self, file_index: dict[str, str]
    ) -> None:
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

    def test_resolve_ts_directory_index(
        self, file_index: dict[str, str]
    ) -> None:
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

    def test_resolve_ts_scoped_external(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="@types/node", names=[], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)
        assert result is None


# ---------------------------------------------------------------------------
# process_imports — Integration
# ---------------------------------------------------------------------------


class TestProcessImportsCreatesRelationships:
    """process_imports creates IMPORTS edges in the graph."""

    def test_process_imports_creates_relationships(
        self, graph: KnowledgeGraph
    ) -> None:
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

    def test_process_imports_relationship_id_format(
        self, graph: KnowledgeGraph
    ) -> None:
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

    def test_process_imports_skips_external(
        self, graph: KnowledgeGraph
    ) -> None:
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

    def test_process_imports_multiple_files(
        self, graph: KnowledgeGraph
    ) -> None:
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

    def test_process_imports_no_duplicates(
        self, graph: KnowledgeGraph
    ) -> None:
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

    def test_process_imports_no_duplicates_across_parse_data(
        self, graph: KnowledgeGraph
    ) -> None:
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


# ---------------------------------------------------------------------------
# resolve_import_path — Python edge cases
# ---------------------------------------------------------------------------


class TestResolvePythonRelativeEdgeCases:
    """Edge cases for Python relative import resolution."""

    def test_single_dot_no_remainder(self) -> None:
        """from . import utils -> resolves to package __init__.py."""
        g = KnowledgeGraph()
        for path in ("src/auth/validate.py", "src/auth/__init__.py"):
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="python",
                )
            )
        index = build_file_index(g)

        imp = ImportInfo(module=".", names=["utils"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, index)
        expected = generate_id(NodeLabel.FILE, "src/auth/__init__.py")
        assert result == expected

    def test_triple_dot_relative(self) -> None:
        """from ...config import settings (3 dots = 2 levels up from parent).

        From src/a/b/c/deep.py: parent=src/a/b/c, up 2=src/a, then config -> src/a/config.py.
        """
        g = KnowledgeGraph()
        for path in (
            "src/a/b/c/deep.py",
            "src/a/config.py",
        ):
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="python",
                )
            )
        index = build_file_index(g)

        imp = ImportInfo(module="...config", names=["settings"], is_relative=True)
        result = resolve_import_path("src/a/b/c/deep.py", imp, index)
        expected = generate_id(NodeLabel.FILE, "src/a/config.py")
        assert result == expected

    def test_relative_with_multi_segment(self) -> None:
        """from ..foo.bar import baz resolves multi-segment remainder.

        From src/pkg/sub/module.py: parent=src/pkg/sub, up 1=src/pkg,
        then foo/bar -> src/pkg/foo/bar.py.
        """
        g = KnowledgeGraph()
        for path in (
            "src/pkg/sub/module.py",
            "src/pkg/foo/bar.py",
        ):
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="python",
                )
            )
        index = build_file_index(g)

        imp = ImportInfo(module="..foo.bar", names=["baz"], is_relative=True)
        result = resolve_import_path("src/pkg/sub/module.py", imp, index)
        expected = generate_id(NodeLabel.FILE, "src/pkg/foo/bar.py")
        assert result == expected

    def test_relative_unresolvable(self) -> None:
        """Relative import targeting a non-existent file returns None."""
        g = KnowledgeGraph()
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

        imp = ImportInfo(module=".nonexistent", names=["x"], is_relative=True)
        result = resolve_import_path("src/app.py", imp, index)
        assert result is None

    def test_absolute_package_init(self) -> None:
        """Absolute import of package resolves to __init__.py."""
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "mypackage/__init__.py"),
                label=NodeLabel.FILE,
                name="__init__.py",
                file_path="mypackage/__init__.py",
                language="python",
            )
        )
        index = build_file_index(g)

        imp = ImportInfo(module="mypackage", names=["thing"], is_relative=False)
        result = resolve_import_path("src/main.py", imp, index)
        expected = generate_id(NodeLabel.FILE, "mypackage/__init__.py")
        assert result == expected

    def test_absolute_deep_dotted(self) -> None:
        """Absolute import with deep dotted path resolves correctly."""
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/core/auth/jwt.py"),
                label=NodeLabel.FILE,
                name="jwt.py",
                file_path="src/core/auth/jwt.py",
                language="python",
            )
        )
        index = build_file_index(g)

        imp = ImportInfo(module="src.core.auth.jwt", names=["decode"], is_relative=False)
        result = resolve_import_path("app/main.py", imp, index)
        expected = generate_id(NodeLabel.FILE, "src/core/auth/jwt.py")
        assert result == expected


# ---------------------------------------------------------------------------
# resolve_import_path — PHP
# ---------------------------------------------------------------------------


def _build_php_index(*paths: str) -> dict[str, str]:
    """Helper to build a file index from PHP file paths."""
    g = KnowledgeGraph()
    for path in paths:
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, path),
                label=NodeLabel.FILE,
                name=path.rsplit("/", 1)[-1],
                file_path=path,
                language="php",
            )
        )
    return build_file_index(g)


class TestResolvePhpNamespace:
    """PHP namespace (use) resolution via PSR-4 path conversion."""

    def test_direct_namespace_path(self) -> None:
        r"""use App\Models\User -> App/Models/User.php."""
        index = _build_php_index("App/Models/User.php")

        imp = ImportInfo(module=r"App\Models\User", names=["User"], is_relative=False)
        result = resolve_import_path("api/contacts.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "App/Models/User.php")
        assert result == expected

    def test_psr4_app_prefix_strip(self) -> None:
        r"""use App\Services\Auth -> app/Services/Auth.php."""
        index = _build_php_index("app/Services/Auth.php")

        imp = ImportInfo(module=r"App\Services\Auth", names=["Auth"], is_relative=False)
        result = resolve_import_path("api/login.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "app/Services/Auth.php")
        assert result == expected

    def test_psr4_src_prefix_strip(self) -> None:
        r"""use App\Util\Helper -> src/Util/Helper.php."""
        index = _build_php_index("src/Util/Helper.php")

        imp = ImportInfo(module=r"App\Util\Helper", names=["Helper"], is_relative=False)
        result = resolve_import_path("api/main.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "src/Util/Helper.php")
        assert result == expected

    def test_psr4_lib_prefix_strip(self) -> None:
        r"""use App\SDK\Client -> lib/SDK/Client.php."""
        index = _build_php_index("lib/SDK/Client.php")

        imp = ImportInfo(module=r"App\SDK\Client", names=["Client"], is_relative=False)
        result = resolve_import_path("api/main.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "lib/SDK/Client.php")
        assert result == expected

    def test_lowercase_first_segment(self) -> None:
        r"""use MyNamespace\Service -> mynamespace/Service.php."""
        index = _build_php_index("mynamespace/Service.php")

        imp = ImportInfo(module=r"MyNamespace\Service", names=["Service"], is_relative=False)
        result = resolve_import_path("api/main.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "mynamespace/Service.php")
        assert result == expected

    def test_deeply_nested_namespace(self) -> None:
        r"""use App\Auth\Services\Token\Manager -> app/Auth/Services/Token/Manager.php."""
        index = _build_php_index("app/Auth/Services/Token/Manager.php")

        imp = ImportInfo(module=r"App\Auth\Services\Token\Manager", names=["Manager"], is_relative=False)
        result = resolve_import_path("api/main.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "app/Auth/Services/Token/Manager.php")
        assert result == expected

    def test_unresolvable_namespace(self) -> None:
        """Unknown namespace returns None."""
        index = _build_php_index("app/Models/User.php")

        imp = ImportInfo(module=r"Vendor\External\Lib", names=["Lib"], is_relative=False)
        result = resolve_import_path("api/main.php", imp, index)
        assert result is None

    def test_single_segment_namespace(self) -> None:
        """Single-segment namespace (no backslash) tries direct path."""
        index = _build_php_index("SomeClass.php")

        imp = ImportInfo(module="SomeClass", names=["SomeClass"], is_relative=False)
        result = resolve_import_path("api/main.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "SomeClass.php")
        assert result == expected

    def test_empty_module_returns_none(self) -> None:
        """Empty module string returns None."""
        index = _build_php_index("app/foo.php")

        imp = ImportInfo(module="", names=[], is_relative=False)
        result = resolve_import_path("api/main.php", imp, index)
        assert result is None


class TestResolvePhpInclude:
    """PHP relative include/require resolution."""

    def test_same_dir_include(self) -> None:
        """require_once 'helpers.php' from api/contacts.php."""
        index = _build_php_index("api/contacts.php", "api/helpers.php")

        imp = ImportInfo(module="helpers.php", names=[], is_relative=True)
        result = resolve_import_path("api/contacts.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "api/helpers.php")
        assert result == expected

    def test_parent_dir_include(self) -> None:
        """require_once '../config/db.php' from api/contacts.php."""
        index = _build_php_index("api/contacts.php", "config/db.php")

        imp = ImportInfo(module="../config/db.php", names=[], is_relative=True)
        result = resolve_import_path("api/contacts.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "config/db.php")
        assert result == expected

    def test_up_two_levels_include(self) -> None:
        """require_once '../../shared/utils.php' from api/v2/handlers/main.php."""
        index = _build_php_index("api/v2/handlers/main.php", "api/shared/utils.php")

        imp = ImportInfo(module="../../shared/utils.php", names=[], is_relative=True)
        result = resolve_import_path("api/v2/handlers/main.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "api/shared/utils.php")
        assert result == expected

    def test_subdirectory_include(self) -> None:
        """require_once 'helpers/format.php' from api/contacts.php."""
        index = _build_php_index("api/contacts.php", "api/helpers/format.php")

        imp = ImportInfo(module="helpers/format.php", names=[], is_relative=True)
        result = resolve_import_path("api/contacts.php", imp, index)
        expected = generate_id(NodeLabel.FILE, "api/helpers/format.php")
        assert result == expected

    def test_unresolvable_include(self) -> None:
        """Include of non-existent file returns None."""
        index = _build_php_index("api/contacts.php")

        imp = ImportInfo(module="nonexistent.php", names=[], is_relative=True)
        result = resolve_import_path("api/contacts.php", imp, index)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_import_path — HTML
# ---------------------------------------------------------------------------


def _build_html_index(*paths: str) -> dict[str, str]:
    """Helper to build a file index for HTML test scenarios."""
    g = KnowledgeGraph()
    for path in paths:
        lang = "html" if path.endswith((".html", ".htm")) else "javascript"
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, path),
                label=NodeLabel.FILE,
                name=path.rsplit("/", 1)[-1],
                file_path=path,
                language=lang,
            )
        )
    return build_file_index(g)


class TestResolveHtmlScript:
    """HTML <script src> resolution."""

    def test_relative_script_src(self) -> None:
        """<script src=\"./app.js\"> from public/index.html."""
        index = _build_html_index("public/index.html", "public/app.js")

        imp = ImportInfo(module="./app.js", names=[], is_relative=True)
        result = resolve_import_path("public/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "public/app.js")
        assert result == expected

    def test_bare_relative_script(self) -> None:
        """<script src=\"assets/vendor/lib.js\"> from public/index.html."""
        index = _build_html_index("public/index.html", "public/assets/vendor/lib.js")

        imp = ImportInfo(module="assets/vendor/lib.js", names=[], is_relative=True)
        result = resolve_import_path("public/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "public/assets/vendor/lib.js")
        assert result == expected

    def test_parent_dir_script(self) -> None:
        """<script src=\"../shared/utils.js\"> from public/pages/about.html."""
        index = _build_html_index("public/pages/about.html", "public/shared/utils.js")

        imp = ImportInfo(module="../shared/utils.js", names=[], is_relative=True)
        result = resolve_import_path("public/pages/about.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "public/shared/utils.js")
        assert result == expected

    def test_cdn_url_returns_none(self) -> None:
        """<script src=\"https://cdn.example.com/lib.js\"> -> None."""
        index = _build_html_index("public/index.html")

        imp = ImportInfo(module="https://cdn.example.com/lib.js", names=[], is_relative=False)
        result = resolve_import_path("public/index.html", imp, index)
        assert result is None

    def test_protocol_relative_returns_none(self) -> None:
        """<script src=\"//cdn.example.com/lib.js\"> -> None."""
        index = _build_html_index("public/index.html")

        imp = ImportInfo(module="//cdn.example.com/lib.js", names=[], is_relative=False)
        result = resolve_import_path("public/index.html", imp, index)
        assert result is None

    def test_extensionless_resolves_with_js(self) -> None:
        """<script src=\"./app\"> resolves to app.js if it exists."""
        index = _build_html_index("public/index.html", "public/app.js")

        imp = ImportInfo(module="./app", names=[], is_relative=True)
        result = resolve_import_path("public/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "public/app.js")
        assert result == expected

    def test_extensionless_resolves_with_ts(self) -> None:
        """<script src=\"./app\"> resolves to app.ts if it exists."""
        index = _build_html_index("public/index.html", "public/app.ts")

        imp = ImportInfo(module="./app", names=[], is_relative=True)
        result = resolve_import_path("public/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "public/app.ts")
        assert result == expected

    def test_empty_module_returns_none(self) -> None:
        """Empty module string returns None."""
        index = _build_html_index("public/index.html")

        imp = ImportInfo(module="", names=[], is_relative=True)
        result = resolve_import_path("public/index.html", imp, index)
        assert result is None

    def test_unresolvable_script_returns_none(self) -> None:
        """Script pointing to non-existent file returns None."""
        index = _build_html_index("public/index.html")

        imp = ImportInfo(module="./missing.js", names=[], is_relative=True)
        result = resolve_import_path("public/index.html", imp, index)
        assert result is None

    def test_query_string_stripped(self) -> None:
        """<script src=\"app.js?v=2.0\"> resolves after stripping query string."""
        index = _build_html_index("dashboard/index.html", "dashboard/app.js")

        imp = ImportInfo(module="app.js?v=2.0", names=[], is_relative=True)
        result = resolve_import_path("dashboard/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "dashboard/app.js")
        assert result == expected

    def test_query_string_with_version_hash(self) -> None:
        """<script src=\"bundle.js?v=20260217b\"> strips complex query strings."""
        index = _build_html_index("dashboard/page.html", "dashboard/bundle.js")

        imp = ImportInfo(module="bundle.js?v=20260217b", names=[], is_relative=True)
        result = resolve_import_path("dashboard/page.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "dashboard/bundle.js")
        assert result == expected

    def test_absolute_path_site_root(self) -> None:
        """/assets/vendor/react.js resolves to repo-relative path."""
        index = _build_html_index(
            "dashboard/index.html",
            "assets/vendor/react.js",
        )

        imp = ImportInfo(module="/assets/vendor/react.js", names=[], is_relative=False)
        result = resolve_import_path("dashboard/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "assets/vendor/react.js")
        assert result == expected

    def test_absolute_path_with_query_string(self) -> None:
        """/assets/app.js?v=3 strips query and resolves from root."""
        index = _build_html_index("dashboard/index.html", "assets/app.js")

        imp = ImportInfo(module="/assets/app.js?v=3", names=[], is_relative=False)
        result = resolve_import_path("dashboard/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "assets/app.js")
        assert result == expected

    def test_css_link_resolves(self) -> None:
        """<link rel=\"stylesheet\" href=\"styles.css\"> resolves as import."""
        index = _build_html_index("dashboard/index.html", "dashboard/styles.css")

        imp = ImportInfo(module="styles.css", names=[], is_relative=True)
        result = resolve_import_path("dashboard/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "dashboard/styles.css")
        assert result == expected

    def test_css_link_absolute_path(self) -> None:
        """/assets/vendor/leaflet.css resolves from repo root."""
        index = _build_html_index(
            "dashboard/index.html",
            "assets/vendor/leaflet.css",
        )

        imp = ImportInfo(module="/assets/vendor/leaflet.css", names=[], is_relative=False)
        result = resolve_import_path("dashboard/index.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "assets/vendor/leaflet.css")
        assert result == expected

    def test_css_link_with_query_string(self) -> None:
        """<link href=\"public.css?v=7\"> strips query string."""
        index = _build_html_index("public/features.html", "public/public.css")

        imp = ImportInfo(module="public.css?v=7", names=[], is_relative=True)
        result = resolve_import_path("public/features.html", imp, index)
        expected = generate_id(NodeLabel.FILE, "public/public.css")
        assert result == expected


# ---------------------------------------------------------------------------
# resolve_import_path — JS/TS edge cases
# ---------------------------------------------------------------------------


class TestResolveJsTsEdgeCases:
    """Additional JS/TS resolution edge cases."""

    def test_ts_parent_dir_import(self) -> None:
        """import from '../utils' resolves correctly."""
        g = KnowledgeGraph()
        for path in ("lib/sub/index.ts", "lib/utils.ts"):
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="typescript",
                )
            )
        index = build_file_index(g)

        imp = ImportInfo(module="../utils", names=["foo"], is_relative=False)
        result = resolve_import_path("lib/sub/index.ts", imp, index)
        expected = generate_id(NodeLabel.FILE, "lib/utils.ts")
        assert result == expected

    def test_js_with_explicit_extension(self) -> None:
        """import './utils.js' with explicit .js extension."""
        g = KnowledgeGraph()
        for path in ("src/index.js", "src/utils.js"):
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="javascript",
                )
            )
        index = build_file_index(g)

        imp = ImportInfo(module="./utils.js", names=["foo"], is_relative=False)
        result = resolve_import_path("src/index.js", imp, index)
        expected = generate_id(NodeLabel.FILE, "src/utils.js")
        assert result == expected

    def test_tsx_extension_resolution(self) -> None:
        """import './Button' resolves to Button.tsx."""
        g = KnowledgeGraph()
        for path in ("src/App.tsx", "src/Button.tsx"):
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="tsx",
                )
            )
        index = build_file_index(g)

        imp = ImportInfo(module="./Button", names=["Button"], is_relative=False)
        result = resolve_import_path("src/App.tsx", imp, index)
        expected = generate_id(NodeLabel.FILE, "src/Button.tsx")
        assert result == expected
