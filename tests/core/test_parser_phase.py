"""Tests for the parsing processor (Phase 3)."""

from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, NodeLabel, RelType, generate_id
from axon.core.ingestion.parser_phase import (
    FileParseData,
    get_parser,
    parse_file,
    process_parsing,
)
from axon.core.ingestion.walker import FileEntry
from axon.core.parsers.python_lang import PythonParser
from axon.core.parsers.typescript import TypeScriptParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph pre-populated with File nodes for test files."""
    g = KnowledgeGraph()

    # Python file node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.FILE, "src/utils.py"),
            label=NodeLabel.FILE,
            name="utils.py",
            file_path="src/utils.py",
            language="python",
        )
    )

    # TypeScript file node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.FILE, "src/app.ts"),
            label=NodeLabel.FILE,
            name="app.ts",
            file_path="src/app.ts",
            language="typescript",
        )
    )

    return g


PYTHON_CODE = """\
class UserService:
    def get_user(self, user_id: int) -> str:
        return str(user_id)

    def delete_user(self, user_id: int) -> None:
        pass

def helper(x: int) -> int:
    return x + 1
"""

TYPESCRIPT_CODE = """\
interface Config {
    host: string;
    port: number;
}

class App {
    start(): void {}
}

function run(config: Config): void {
    const app = new App();
    app.start();
}
"""

JAVASCRIPT_CODE = """\
function add(a, b) {
    return a + b;
}
"""


def _make_file_entry(
    path: str, content: str, language: str
) -> FileEntry:
    return FileEntry(path=path, content=content, language=language)


# ---------------------------------------------------------------------------
# get_parser tests
# ---------------------------------------------------------------------------


class TestGetParserPython:
    """get_parser returns PythonParser for 'python'."""

    def test_get_parser_python(self) -> None:
        parser = get_parser("python")
        assert isinstance(parser, PythonParser)


class TestGetParserTypeScript:
    """get_parser returns TypeScriptParser for 'typescript'."""

    def test_get_parser_typescript(self) -> None:
        parser = get_parser("typescript")
        assert isinstance(parser, TypeScriptParser)
        assert parser.dialect == "typescript"


class TestGetParserJavaScript:
    """get_parser returns TypeScriptParser with 'javascript' dialect."""

    def test_get_parser_javascript(self) -> None:
        parser = get_parser("javascript")
        assert isinstance(parser, TypeScriptParser)
        assert parser.dialect == "javascript"


class TestGetParserUnsupported:
    """get_parser raises ValueError for unknown languages."""

    def test_get_parser_unsupported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            get_parser("rust")


# ---------------------------------------------------------------------------
# parse_file tests
# ---------------------------------------------------------------------------


class TestParseFilePython:
    """parse_file parses Python source and returns correct symbols."""

    def test_parse_file_python(self) -> None:
        data = parse_file("src/utils.py", PYTHON_CODE, "python")

        assert isinstance(data, FileParseData)
        assert data.file_path == "src/utils.py"
        assert data.language == "python"

        symbol_names = [s.name for s in data.parse_result.symbols]
        assert "UserService" in symbol_names
        assert "get_user" in symbol_names
        assert "delete_user" in symbol_names
        assert "helper" in symbol_names

    def test_method_has_class_name(self) -> None:
        data = parse_file("src/utils.py", PYTHON_CODE, "python")
        methods = [s for s in data.parse_result.symbols if s.kind == "method"]
        for m in methods:
            assert m.class_name == "UserService"


class TestParseFileTypeScript:
    """parse_file parses TypeScript source and returns correct symbols."""

    def test_parse_file_typescript(self) -> None:
        data = parse_file("src/app.ts", TYPESCRIPT_CODE, "typescript")

        assert isinstance(data, FileParseData)
        assert data.file_path == "src/app.ts"
        assert data.language == "typescript"

        symbol_names = [s.name for s in data.parse_result.symbols]
        assert "Config" in symbol_names
        assert "App" in symbol_names
        assert "run" in symbol_names


# ---------------------------------------------------------------------------
# process_parsing tests
# ---------------------------------------------------------------------------


class TestProcessParsingCreatesFunctionNodes:
    """process_parsing creates Function nodes in the graph."""

    def test_process_parsing_creates_function_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        func_nodes = graph.get_nodes_by_label(NodeLabel.FUNCTION)
        func_names = {n.name for n in func_nodes}
        assert "helper" in func_names

    def test_function_node_properties(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        func_id = generate_id(NodeLabel.FUNCTION, "src/utils.py", "helper")
        node = graph.get_node(func_id)
        assert node is not None
        assert node.name == "helper"
        assert node.file_path == "src/utils.py"
        assert node.start_line > 0
        assert node.end_line >= node.start_line
        assert "def helper" in node.content
        assert node.signature != ""


class TestProcessParsingCreatesClassNodes:
    """process_parsing creates Class nodes in the graph."""

    def test_process_parsing_creates_class_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        class_nodes = graph.get_nodes_by_label(NodeLabel.CLASS)
        class_names = {n.name for n in class_nodes}
        assert "UserService" in class_names

    def test_class_node_has_content(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        class_id = generate_id(NodeLabel.CLASS, "src/utils.py", "UserService")
        node = graph.get_node(class_id)
        assert node is not None
        assert "class UserService" in node.content


class TestProcessParsingCreatesMethodNodes:
    """process_parsing creates Method nodes with class_name set."""

    def test_process_parsing_creates_method_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        method_names = {n.name for n in method_nodes}
        assert "get_user" in method_names
        assert "delete_user" in method_names

    def test_method_nodes_have_class_name(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        for method in method_nodes:
            assert method.class_name == "UserService"

    def test_method_node_id_uses_class_dot_method(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_id = generate_id(
            NodeLabel.METHOD, "src/utils.py", "UserService.get_user"
        )
        node = graph.get_node(method_id)
        assert node is not None
        assert node.name == "get_user"


class TestProcessParsingCreatesDefinesRelationships:
    """process_parsing creates DEFINES relationships from File to Symbol."""

    def test_process_parsing_creates_defines_relationships(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        assert len(defines_rels) > 0

        file_id = generate_id(NodeLabel.FILE, "src/utils.py")
        # All DEFINES relationships should originate from the file node.
        for rel in defines_rels:
            assert rel.source == file_id

    def test_defines_relationship_targets_symbol(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        target_ids = {rel.target for rel in defines_rels}

        # The function node should be a target.
        func_id = generate_id(NodeLabel.FUNCTION, "src/utils.py", "helper")
        assert func_id in target_ids

        # The class node should be a target.
        class_id = generate_id(NodeLabel.CLASS, "src/utils.py", "UserService")
        assert class_id in target_ids

    def test_defines_relationship_id_format(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        for rel in defines_rels:
            assert rel.id.startswith("defines:")
            assert "->" in rel.id


class TestProcessParsingReturnsParseData:
    """process_parsing returns FileParseData for use by later phases."""

    def test_process_parsing_returns_parse_data(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [
            _make_file_entry("src/utils.py", PYTHON_CODE, "python"),
            _make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript"),
        ]
        result = process_parsing(files, graph)

        assert len(result) == 2
        assert all(isinstance(d, FileParseData) for d in result)

    def test_parse_data_carries_imports(
        self, graph: KnowledgeGraph
    ) -> None:
        code_with_import = "import os\n\ndef main():\n    pass\n"
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/main.py"),
                label=NodeLabel.FILE,
                name="main.py",
                file_path="src/main.py",
                language="python",
            )
        )
        files = [_make_file_entry("src/main.py", code_with_import, "python")]
        result = process_parsing(files, graph)

        assert len(result[0].parse_result.imports) > 0

    def test_parse_data_carries_calls(
        self, graph: KnowledgeGraph
    ) -> None:
        code_with_call = "def foo():\n    bar()\n"
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/caller.py"),
                label=NodeLabel.FILE,
                name="caller.py",
                file_path="src/caller.py",
                language="python",
            )
        )
        files = [_make_file_entry("src/caller.py", code_with_call, "python")]
        result = process_parsing(files, graph)

        call_names = [c.name for c in result[0].parse_result.calls]
        assert "bar" in call_names


class TestProcessParsingHandlesError:
    """process_parsing handles bad content gracefully without crashing."""

    def test_process_parsing_handles_error(
        self, graph: KnowledgeGraph
    ) -> None:
        # Provide an unsupported language to trigger the error path.
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/bad.rs"),
                label=NodeLabel.FILE,
                name="bad.rs",
                file_path="src/bad.rs",
                language="rust",
            )
        )
        files = [_make_file_entry("src/bad.rs", "fn main() {}", "rust")]
        result = process_parsing(files, graph)

        # Should still return a FileParseData with empty result.
        assert len(result) == 1
        assert result[0].parse_result.symbols == []
        assert result[0].parse_result.imports == []

    def test_error_does_not_affect_other_files(
        self, graph: KnowledgeGraph
    ) -> None:
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/bad.rs"),
                label=NodeLabel.FILE,
                name="bad.rs",
                file_path="src/bad.rs",
                language="rust",
            )
        )
        files = [
            _make_file_entry("src/bad.rs", "fn main() {}", "rust"),
            _make_file_entry("src/utils.py", PYTHON_CODE, "python"),
        ]
        result = process_parsing(files, graph)

        assert len(result) == 2
        # The Rust file should have empty symbols.
        assert result[0].parse_result.symbols == []
        # The Python file should parse successfully.
        assert len(result[1].parse_result.symbols) > 0


class TestProcessParsingTypeScript:
    """process_parsing handles TypeScript interface and class nodes."""

    def test_creates_interface_nodes(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript")]
        process_parsing(files, graph)

        iface_nodes = graph.get_nodes_by_label(NodeLabel.INTERFACE)
        iface_names = {n.name for n in iface_nodes}
        assert "Config" in iface_names

    def test_creates_ts_class_and_method_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript")]
        process_parsing(files, graph)

        class_nodes = graph.get_nodes_by_label(NodeLabel.CLASS)
        class_names = {n.name for n in class_nodes}
        assert "App" in class_names

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        method_names = {n.name for n in method_nodes}
        assert "start" in method_names
