"""Tests for Phase 9: traits, enums, serialization methods, framework exemptions.

Four improvement categories:
1. PHP traits — trait declarations parsed as class-like, `use Trait;` as heritage
2. TypeScript enum declarations — parsed as enum symbols
3. Implicit interface methods — serialization/iterator methods exempt from dead code
4. Framework directory/method exemptions — migration up/down, command handle, etc.
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
from axon.core.ingestion.dead_code import (
    _is_example_file,
    _is_exempt,
    _is_framework_method,
    process_dead_code,
)
from axon.core.parsers.base import ParseResult, SymbolInfo
from axon.core.parsers.php import PhpParser
from axon.core.parsers.typescript import TypeScriptParser


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
# 1. PHP traits
# ---------------------------------------------------------------------------


class TestPhpTraits:
    """PHP trait declarations and `use Trait;` heritage extraction."""

    def test_trait_extracted_as_class_symbol(self) -> None:
        parser = PhpParser()
        code = """<?php
trait Loggable {
    public function log($msg) { echo $msg; }
}
"""
        result = parser.parse(code, "traits.php")
        trait_syms = [s for s in result.symbols if s.name == "Loggable"]
        assert len(trait_syms) == 1
        assert trait_syms[0].kind == "class"

    def test_trait_methods_extracted(self) -> None:
        parser = PhpParser()
        code = """<?php
trait Cacheable {
    public function cacheKey() { return 'key'; }
    public function clearCache() {}
}
"""
        result = parser.parse(code, "cacheable.php")
        methods = [s for s in result.symbols if s.kind == "method"]
        method_names = {m.name for m in methods}
        assert "cacheKey" in method_names
        assert "clearCache" in method_names
        # Methods should have class_name set to the trait name
        for m in methods:
            assert m.class_name == "Cacheable"

    def test_use_trait_creates_heritage(self) -> None:
        parser = PhpParser()
        code = """<?php
trait Loggable {}
trait Cacheable {}

class UserService {
    use Loggable, Cacheable;

    public function getUser() { return null; }
}
"""
        result = parser.parse(code, "service.php")
        heritage = result.heritage
        # UserService should extend both Loggable and Cacheable (via use)
        assert ("UserService", "extends", "Loggable") in heritage
        assert ("UserService", "extends", "Cacheable") in heritage

    def test_trait_use_with_qualified_name(self) -> None:
        parser = PhpParser()
        code = r"""<?php
class Foo {
    use App\Traits\Auditable;
}
"""
        result = parser.parse(code, "foo.php")
        heritage_names = [h[2] for h in result.heritage]
        # Should use short name
        assert "Auditable" in heritage_names


# ---------------------------------------------------------------------------
# 2. TypeScript enum declarations
# ---------------------------------------------------------------------------


class TestTypeScriptEnums:
    """TypeScript enum declarations parsed as enum symbols."""

    def test_basic_enum(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        code = """
enum Direction {
    Up,
    Down,
    Left,
    Right
}
"""
        result = parser.parse(code, "direction.ts")
        enum_syms = [s for s in result.symbols if s.name == "Direction"]
        assert len(enum_syms) == 1
        assert enum_syms[0].kind == "enum"

    def test_string_enum(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        code = """
enum Color {
    Red = "RED",
    Green = "GREEN",
    Blue = "BLUE"
}
"""
        result = parser.parse(code, "color.ts")
        enum_syms = [s for s in result.symbols if s.name == "Color"]
        assert len(enum_syms) == 1
        assert enum_syms[0].kind == "enum"

    def test_const_enum(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        code = """
const enum Status {
    Active = 1,
    Inactive = 0
}
"""
        result = parser.parse(code, "status.ts")
        enum_syms = [s for s in result.symbols if s.name == "Status"]
        assert len(enum_syms) == 1
        assert enum_syms[0].kind == "enum"

    def test_enum_line_numbers(self) -> None:
        parser = TypeScriptParser(dialect="typescript")
        code = """// header
enum Fruit {
    Apple,
    Banana
}
"""
        result = parser.parse(code, "fruit.ts")
        enum_sym = [s for s in result.symbols if s.name == "Fruit"][0]
        assert enum_sym.start_line == 2
        assert enum_sym.end_line == 5


# ---------------------------------------------------------------------------
# 3. Implicit interface methods
# ---------------------------------------------------------------------------


class TestImplicitInterfaceMethods:
    """Methods like jsonSerialize, toArray, etc. are exempt from dead code."""

    @pytest.mark.parametrize("method_name", [
        "jsonSerialize", "count", "offsetGet", "offsetSet",
        "offsetExists", "offsetUnset", "getIterator", "current",
        "key", "next", "rewind", "valid", "serialize", "unserialize",
        "toArray", "toJson", "toJSON",
    ])
    def test_implicit_method_not_flagged_dead(self, method_name: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/Model.php")
        _add_symbol_node(
            g, NodeLabel.CLASS, "app/Model.php", "Model", 1, 50,
        )
        node_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/Model.php", method_name, 10, 15,
            class_name="Model",
        )

        dead_count = process_dead_code(g)
        node = g.get_node(node_id)
        assert node is not None
        assert not node.is_dead

    def test_regular_method_still_flagged_dead(self) -> None:
        """A non-implicit method with no calls IS flagged dead."""
        g = KnowledgeGraph()
        _add_file_node(g, "app/Model.php")
        _add_symbol_node(
            g, NodeLabel.CLASS, "app/Model.php", "Model", 1, 50,
        )
        node_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/Model.php", "calculateTotal", 10, 15,
            class_name="Model",
        )

        dead_count = process_dead_code(g)
        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead


# ---------------------------------------------------------------------------
# 4. Framework directory/method exemptions
# ---------------------------------------------------------------------------


class TestFrameworkMethodExemptions:
    """Framework-invoked methods in specific directories are exempt."""

    def test_is_framework_method_migrations(self) -> None:
        assert _is_framework_method("up", "database/migrations/2024_create_users.php")
        assert _is_framework_method("down", "database/migrations/2024_create_users.php")

    def test_is_framework_method_seeders(self) -> None:
        assert _is_framework_method("run", "database/seeders/UserSeeder.php")
        assert _is_framework_method("run", "database/seeds/UserSeeder.php")

    def test_is_framework_method_commands(self) -> None:
        assert _is_framework_method("handle", "app/Console/Commands/ImportData.php")

    def test_is_framework_method_listeners(self) -> None:
        assert _is_framework_method("handle", "app/Listeners/SendNotification.php")

    def test_is_framework_method_providers(self) -> None:
        assert _is_framework_method("register", "app/Providers/AppServiceProvider.php")
        assert _is_framework_method("boot", "app/Providers/AppServiceProvider.php")

    def test_is_framework_method_middleware(self) -> None:
        assert _is_framework_method("handle", "app/Middleware/AuthCheck.php")

    def test_is_framework_method_jobs(self) -> None:
        assert _is_framework_method("handle", "app/Jobs/ProcessPayment.php")

    def test_non_framework_dir_not_exempt(self) -> None:
        assert not _is_framework_method("handle", "app/Services/UserService.php")
        assert not _is_framework_method("up", "app/Models/User.php")

    def test_non_framework_method_not_exempt(self) -> None:
        assert not _is_framework_method("calculate", "database/migrations/foo.php")

    def test_migration_up_not_flagged_dead(self) -> None:
        """End-to-end: migration up() method is not flagged as dead."""
        g = KnowledgeGraph()
        path = "database/migrations/2024_create_users.php"
        _add_file_node(g, path)
        _add_symbol_node(
            g, NodeLabel.CLASS, path, "CreateUsers", 1, 30,
        )
        up_id = _add_symbol_node(
            g, NodeLabel.METHOD, path, "up", 5, 15,
            class_name="CreateUsers",
        )
        down_id = _add_symbol_node(
            g, NodeLabel.METHOD, path, "down", 17, 25,
            class_name="CreateUsers",
        )

        dead_count = process_dead_code(g)
        assert not g.get_node(up_id).is_dead
        assert not g.get_node(down_id).is_dead

    def test_command_handle_not_flagged_dead(self) -> None:
        """End-to-end: command handle() method is not flagged as dead."""
        g = KnowledgeGraph()
        path = "app/Console/Commands/ImportData.php"
        _add_file_node(g, path)
        _add_symbol_node(
            g, NodeLabel.CLASS, path, "ImportData", 1, 50,
        )
        handle_id = _add_symbol_node(
            g, NodeLabel.METHOD, path, "handle", 10, 40,
            class_name="ImportData",
        )

        dead_count = process_dead_code(g)
        assert not g.get_node(handle_id).is_dead

    def test_is_exempt_integrates_framework_method(self) -> None:
        """_is_exempt returns True for framework methods."""
        assert _is_exempt(
            "handle", is_entry_point=False, is_exported=False,
            file_path="app/Console/Commands/ImportData.php",
        )
        assert _is_exempt(
            "up", is_entry_point=False, is_exported=False,
            file_path="database/migrations/2024_create_users.php",
        )
        # Non-framework should not be exempt
        assert not _is_exempt(
            "calculate", is_entry_point=False, is_exported=False,
            file_path="app/Services/Calculator.php",
        )
