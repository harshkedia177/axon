"""Tests for the dead code detection phase (Phase 10)."""

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
from axon.core.ingestion.dead_code import process_dead_code

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
    *,
    is_entry_point: bool = False,
    is_exported: bool = False,
    class_name: str = "",
) -> str:
    """Add a symbol node and return its ID."""
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
            class_name=class_name,
            is_entry_point=is_entry_point,
            is_exported=is_exported,
        )
    )
    return node_id


def _add_calls_relationship(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> None:
    """Add a CALLS relationship from *source_id* to *target_id*."""
    rel_id = f"calls:{source_id}->{target_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source_id,
            target=target_id,
        )
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the test fixture specification.

    - Function:src/main.py:main         (entry point, no incoming calls)
    - Function:src/auth.py:validate     (has incoming calls from main)
    - Function:src/auth.py:unused_helper (no calls, not entry point) -> DEAD
    - Method:src/models.py:User.__init__ (no calls, constructor)    -> NOT dead
    - Function:src/tests/test_auth.py:test_validate (test function) -> NOT dead
    - Function:src/utils.py:orphan_function (no calls, not entry)   -> DEAD
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/main.py")
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/models.py")
    _add_file_node(g, "src/tests/test_auth.py")
    _add_file_node(g, "src/utils.py")

    # Symbols
    main_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/main.py", "main", is_entry_point=True
    )
    validate_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/auth.py", "validate"
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/auth.py", "unused_helper"
    )
    _add_symbol_node(
        g,
        NodeLabel.METHOD,
        "src/models.py",
        "__init__",
        class_name="User",
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/tests/test_auth.py", "test_validate"
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/utils.py", "orphan_function"
    )

    # CALLS: main -> validate
    _add_calls_relationship(g, main_id, validate_id)

    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectsUnusedFunction:
    """Unused helper functions with no incoming calls are flagged as dead."""

    def test_detects_unused_function(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        unused_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "unused_helper"
        )
        node = graph.get_node(unused_id)
        assert node is not None
        assert node.is_dead is True


class TestSkipsEntryPoints:
    """Entry points are never flagged as dead, even without incoming calls."""

    def test_skips_entry_points(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        main_id = generate_id(NodeLabel.FUNCTION, "src/main.py", "main")
        node = graph.get_node(main_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsCalledFunctions:
    """Functions with incoming CALLS relationships are not flagged."""

    def test_skips_called_functions(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        validate_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        node = graph.get_node(validate_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsConstructors:
    """__init__ and __new__ methods are never flagged as dead."""

    def test_skips_constructors(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        init_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__init__"
        )
        node = graph.get_node(init_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsTestFunctions:
    """Test functions (test_*) are never flagged as dead."""

    def test_skips_test_functions(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        test_id = generate_id(
            NodeLabel.FUNCTION, "src/tests/test_auth.py", "test_validate"
        )
        node = graph.get_node(test_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsDunderMethods:
    """Dunder methods (__str__, __repr__, etc.) are never flagged as dead."""

    def test_skips_dunder_methods(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "__str__",
            class_name="User",
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "__repr__",
            class_name="User",
        )

        process_dead_code(g)

        str_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__str__"
        )
        repr_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__repr__"
        )

        str_node = g.get_node(str_id)
        repr_node = g.get_node(repr_id)

        assert str_node is not None
        assert str_node.is_dead is False

        assert repr_node is not None
        assert repr_node.is_dead is False


class TestReturnsCount:
    """process_dead_code returns the correct count of dead symbols."""

    def test_returns_count(self, graph: KnowledgeGraph) -> None:
        count = process_dead_code(graph)

        # unused_helper and orphan_function are the two dead symbols.
        assert count == 2


class TestEmptyGraph:
    """An empty graph produces zero dead symbols."""

    def test_empty_graph(self) -> None:
        g = KnowledgeGraph()
        count = process_dead_code(g)
        assert count == 0


# ---------------------------------------------------------------------------
# Helpers for USES_TYPE and EXTENDS relationships
# ---------------------------------------------------------------------------


def _add_uses_type_relationship(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> None:
    """Add a USES_TYPE relationship from *source_id* to *target_id*."""
    rel_id = f"uses_type:{source_id}->{target_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.USES_TYPE,
            source=source_id,
            target=target_id,
        )
    )


# ---------------------------------------------------------------------------
# USES_TYPE tests
# ---------------------------------------------------------------------------


class TestSkipsTypeReferencedClasses:
    """Classes with incoming USES_TYPE edges are not flagged as dead."""

    def test_class_with_uses_type_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_file_node(g, "src/handler.py")

        class_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/models.py", "Status"
        )
        func_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/handler.py", "handle",
            is_entry_point=True,
        )
        _add_uses_type_relationship(g, func_id, class_id)

        process_dead_code(g)

        node = g.get_node(class_id)
        assert node is not None
        assert node.is_dead is False

    def test_function_with_only_uses_type_still_dead(self) -> None:
        """Functions referenced only as types ARE dead (not classes)."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        _add_file_node(g, "src/handler.py")

        func_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "unused_func"
        )
        other_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/handler.py", "handle",
            is_entry_point=True,
        )
        _add_uses_type_relationship(g, other_id, func_id)

        process_dead_code(g)

        node = g.get_node(func_id)
        assert node is not None
        assert node.is_dead is True


# ---------------------------------------------------------------------------
# Framework decorator tests
# ---------------------------------------------------------------------------


class TestSkipsFrameworkDecoratedFunctions:
    """Functions with framework-registration decorators are not flagged dead."""

    def test_framework_decorated_function_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/server.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/server.py", "list_tools"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["server.list_tools"]

        process_dead_code(g)

        assert node.is_dead is False

    def test_simple_decorator_still_dead(self) -> None:
        """Decorators without dots (e.g., @staticmethod) do not exempt."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "unused"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["staticmethod"]

        process_dead_code(g)

        assert node.is_dead is True

    def test_typing_overload_decorator_exempts(self) -> None:
        """@typing.overload stubs are not dead — they define type signatures."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "overloaded"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["typing.overload"]

        process_dead_code(g)

        assert node.is_dead is False

    def test_functools_wraps_still_dead(self) -> None:
        """Known non-framework dotted decorators (functools.wraps) don't exempt."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "wrapper"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["functools.wraps"]

        process_dead_code(g)

        assert node.is_dead is True


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Methods on classes structurally conforming to a Protocol are not dead."""

    def test_conforming_class_methods_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/base.py")
        _add_file_node(g, "src/impl.py")
        _add_file_node(g, "src/main.py")

        # Protocol class with is_protocol annotation
        proto_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/base.py", "StorageBackend"
        )
        proto_node = g.get_node(proto_id)
        assert proto_node is not None
        proto_node.properties["is_protocol"] = True

        # Protocol methods
        proto_init_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "initialize",
            class_name="StorageBackend",
        )
        proto_close_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "close",
            class_name="StorageBackend",
        )

        # Concrete class structurally conforming (has both methods)
        _add_symbol_node(
            g, NodeLabel.CLASS, "src/impl.py", "KuzuBackend"
        )
        impl_init_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/impl.py", "initialize",
            class_name="KuzuBackend",
        )
        impl_close_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/impl.py", "close",
            class_name="KuzuBackend",
        )

        # A caller calls StorageBackend.initialize (not KuzuBackend)
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/main.py", "main",
            is_entry_point=True,
        )
        _add_calls_relationship(g, caller_id, proto_init_id)
        _add_calls_relationship(g, caller_id, proto_close_id)

        process_dead_code(g)

        # Protocol methods are alive (have incoming CALLS)
        assert g.get_node(proto_init_id).is_dead is False
        assert g.get_node(proto_close_id).is_dead is False

        # Concrete methods should be un-flagged by Protocol conformance
        assert g.get_node(impl_init_id).is_dead is False
        assert g.get_node(impl_close_id).is_dead is False

    def test_non_conforming_class_still_dead(self) -> None:
        """A class with only some protocol methods is still flagged dead."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/base.py")
        _add_file_node(g, "src/partial.py")

        proto_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/base.py", "Backend"
        )
        proto_node = g.get_node(proto_id)
        assert proto_node is not None
        proto_node.properties["is_protocol"] = True

        _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "initialize",
            class_name="Backend",
        )
        _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "close",
            class_name="Backend",
        )

        # Partial class only has "initialize", not "close"
        _add_symbol_node(
            g, NodeLabel.CLASS, "src/partial.py", "Partial"
        )
        partial_method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/partial.py", "initialize",
            class_name="Partial",
        )

        process_dead_code(g)

        partial_method = g.get_node(partial_method_id)
        assert partial_method is not None
        assert partial_method.is_dead is True
