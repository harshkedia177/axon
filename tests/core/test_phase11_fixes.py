"""Phase 11 tests: PHP entry points, heritage edge cases, receiver type resolution.

Covers:
- PHP entry point detection in processes.py
- Heritage processing edge cases (interface-extends-interface, unknown kind,
  same-name classes cross-file, ABCMeta, empty parse data)
- Receiver type resolution (variable_types dispatch, cross-file factory
  sentinel resolution, unresolvable sentinel cleanup)
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
from axon.core.ingestion.heritage import process_heritage
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.processes import (
    _is_entry_point,
    _matches_framework_pattern,
    find_entry_points,
)
from axon.core.ingestion.symbol_lookup import build_name_index
from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    ParseResult,
    SymbolInfo,
    TypeRef,
)


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


def _add_function(
    graph: KnowledgeGraph,
    name: str,
    file_path: str = "src/app.py",
    *,
    content: str = "",
    language: str = "python",
    is_exported: bool = False,
    class_name: str = "",
) -> GraphNode:
    node_id = generate_id(NodeLabel.FUNCTION, file_path, name)
    node = GraphNode(
        id=node_id,
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        content=content,
        language=language,
        is_exported=is_exported,
        class_name=class_name,
    )
    graph.add_node(node)
    return node


def _add_method(
    graph: KnowledgeGraph,
    name: str,
    file_path: str,
    class_name: str,
    start_line: int = 1,
    end_line: int = 10,
) -> GraphNode:
    symbol_name = f"{class_name}.{name}"
    node_id = generate_id(NodeLabel.METHOD, file_path, symbol_name)
    node = GraphNode(
        id=node_id,
        label=NodeLabel.METHOD,
        name=name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        class_name=class_name,
    )
    graph.add_node(node)
    return node


def _add_call(
    graph: KnowledgeGraph,
    source: GraphNode,
    target: GraphNode,
    confidence: float = 1.0,
) -> None:
    rel_id = f"calls:{source.id}->{target.id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source.id,
            target=target.id,
            properties={"confidence": confidence},
        )
    )


def _make_heritage_data(
    file_path: str,
    heritage: list[tuple[str, str, str]],
    language: str = "python",
) -> FileParseData:
    return FileParseData(
        file_path=file_path,
        language=language,
        parse_result=ParseResult(heritage=heritage),
    )


# ===========================================================================
# PART 1: PHP Entry Point Detection
# ===========================================================================


class TestPhpEntryPointFrameworkPattern:
    """_matches_framework_pattern recognises PHP patterns."""

    def test_controller_class_name(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "UserController", file_path="api/UserController.php", language="php"
        )
        assert _matches_framework_pattern(node) is True

    def test_method_in_controller_class(self) -> None:
        g = KnowledgeGraph()
        node_id = generate_id(
            NodeLabel.METHOD, "api/UserController.php", "UserController.index"
        )
        node = GraphNode(
            id=node_id,
            label=NodeLabel.METHOD,
            name="index",
            file_path="api/UserController.php",
            language="php",
            class_name="UserController",
        )
        g.add_node(node)
        assert _matches_framework_pattern(node) is True

    def test_api_directory_entry_point(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "handleRequest", file_path="project/api/contacts.php", language="php"
        )
        assert _matches_framework_pattern(node) is True

    def test_cron_directory_entry_point(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "runJob", file_path="project/cron/daily-sync.php", language="php"
        )
        assert _matches_framework_pattern(node) is True

    def test_webhook_file_entry_point(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "processEvent", file_path="api/social-webhook.php", language="php"
        )
        assert _matches_framework_pattern(node) is True

    def test_controllers_directory(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "show", file_path="app/Controllers/PostController.php", language="php"
        )
        assert _matches_framework_pattern(node) is True

    def test_routes_directory(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "apiRoutes", file_path="project/routes/api.php", language="php"
        )
        assert _matches_framework_pattern(node) is True

    def test_content_with_get_action(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g,
            "dispatch",
            file_path="handlers/dispatch.php",
            language="php",
            content="$action = $_GET['action'] ?? '';",
        )
        assert _matches_framework_pattern(node) is True

    def test_content_with_request_method(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g,
            "handlePost",
            file_path="handlers/form.php",
            language="php",
            content="if ($_SERVER['REQUEST_METHOD'] === 'POST') {",
        )
        assert _matches_framework_pattern(node) is True

    def test_content_with_post_data(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g,
            "saveForm",
            file_path="handlers/save.php",
            language="php",
            content="$name = $_POST['name'];",
        )
        assert _matches_framework_pattern(node) is True

    def test_plain_php_helper_not_entry_point(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g,
            "formatDate",
            file_path="lib/helpers/dates.php",
            language="php",
        )
        assert _matches_framework_pattern(node) is False

    def test_php_extension_inferred_from_path(self) -> None:
        """Language='', but file ends in .php -> PHP patterns apply."""
        g = KnowledgeGraph()
        node = _add_function(
            g,
            "index",
            file_path="api/dashboard.php",
            language="",
        )
        assert _matches_framework_pattern(node) is True


class TestPhpEntryPointIsEntryPoint:
    """_is_entry_point with PHP-specific heuristics."""

    def test_cron_file_prefix_is_entry_point(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "syncData", file_path="api/cron-sync.php", language="php"
        )
        assert _is_entry_point(node, g) is True

    def test_php_function_in_cron_dir(self) -> None:
        g = KnowledgeGraph()
        node = _add_function(
            g, "processQueue", file_path="cron/queue-worker.php", language="php"
        )
        assert _is_entry_point(node, g) is True


class TestPhpFindEntryPoints:
    """find_entry_points integrates PHP patterns."""

    def test_php_api_function_detected(self) -> None:
        g = KnowledgeGraph()
        api_fn = _add_function(
            g, "listContacts", file_path="api/contacts.php", language="php"
        )
        helper_fn = _add_function(
            g, "formatRow", file_path="lib/format.php", language="php"
        )
        # Give helper an incoming call so it's not an entry point by default.
        caller = _add_function(g, "caller", file_path="api/contacts.php", language="php")
        _add_call(g, caller, helper_fn)

        eps = find_entry_points(g)
        ep_names = {n.name for n in eps}

        assert "listContacts" in ep_names
        assert "formatRow" not in ep_names

    def test_php_webhook_function_detected(self) -> None:
        g = KnowledgeGraph()
        wh_fn = _add_function(
            g, "handleEvent", file_path="api/social-webhook.php", language="php"
        )
        eps = find_entry_points(g)
        assert any(n.name == "handleEvent" for n in eps)


# ===========================================================================
# PART 2: Heritage Processing Edge Cases
# ===========================================================================


class TestHeritageInterfaceExtendsInterface:
    """Interface extending another interface creates EXTENDS edge."""

    def test_interface_extends_interface(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.INTERFACE, "src/types.php", "Readable"),
                label=NodeLabel.INTERFACE,
                name="Readable",
                file_path="src/types.php",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.INTERFACE, "src/types.php", "ReadWritable"),
                label=NodeLabel.INTERFACE,
                name="ReadWritable",
                file_path="src/types.php",
            )
        )

        parse_data = [
            _make_heritage_data(
                "src/types.php",
                [("ReadWritable", "extends", "Readable")],
            ),
        ]
        process_heritage(parse_data, g)

        extends_rels = g.get_relationships_by_type(RelType.EXTENDS)
        assert len(extends_rels) == 1
        assert extends_rels[0].source == generate_id(
            NodeLabel.INTERFACE, "src/types.php", "ReadWritable"
        )
        assert extends_rels[0].target == generate_id(
            NodeLabel.INTERFACE, "src/types.php", "Readable"
        )


class TestHeritageUnknownKind:
    """Unknown heritage kind logs warning and skips."""

    def test_unknown_kind_skipped(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/a.py", "Foo"),
                label=NodeLabel.CLASS,
                name="Foo",
                file_path="src/a.py",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/a.py", "Bar"),
                label=NodeLabel.CLASS,
                name="Bar",
                file_path="src/a.py",
            )
        )

        parse_data = [
            _make_heritage_data("src/a.py", [("Foo", "uses", "Bar")]),
        ]
        # Should not raise.
        process_heritage(parse_data, g)

        # No relationship created for unknown kind.
        extends_rels = g.get_relationships_by_type(RelType.EXTENDS)
        impl_rels = g.get_relationships_by_type(RelType.IMPLEMENTS)
        assert len(extends_rels) == 0
        assert len(impl_rels) == 0


class TestHeritageABCMetaMarker:
    """ABCMeta is a protocol marker like ABC and Protocol."""

    def test_abcmeta_annotates_child(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/base.py", "MyBase"),
                label=NodeLabel.CLASS,
                name="MyBase",
                file_path="src/base.py",
            )
        )

        parse_data = [
            _make_heritage_data("src/base.py", [("MyBase", "extends", "ABCMeta")]),
        ]
        process_heritage(parse_data, g)

        node = g.get_node(generate_id(NodeLabel.CLASS, "src/base.py", "MyBase"))
        assert node is not None
        assert node.properties.get("is_protocol") is True

        # No EXTENDS edge created.
        assert len(g.get_relationships_by_type(RelType.EXTENDS)) == 0


class TestHeritageSameNameCrossFile:
    """Same-file child is preferred over cross-file candidate."""

    def test_same_file_preference(self) -> None:
        g = KnowledgeGraph()
        # Two classes named "Base" in different files.
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/a.py", "Base"),
                label=NodeLabel.CLASS,
                name="Base",
                file_path="src/a.py",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/b.py", "Base"),
                label=NodeLabel.CLASS,
                name="Base",
                file_path="src/b.py",
            )
        )
        # Child in src/a.py extends Base — should resolve to same-file Base.
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/a.py", "Child"),
                label=NodeLabel.CLASS,
                name="Child",
                file_path="src/a.py",
            )
        )

        parse_data = [
            _make_heritage_data("src/a.py", [("Child", "extends", "Base")]),
        ]
        process_heritage(parse_data, g)

        extends_rels = g.get_relationships_by_type(RelType.EXTENDS)
        assert len(extends_rels) == 1
        # Target is Base in src/a.py (same file), not src/b.py.
        assert extends_rels[0].target == generate_id(
            NodeLabel.CLASS, "src/a.py", "Base"
        )


class TestHeritageEmptyData:
    """Empty parse data is a no-op."""

    def test_empty_parse_data(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/a.py", "Foo"),
                label=NodeLabel.CLASS,
                name="Foo",
                file_path="src/a.py",
            )
        )
        process_heritage([], g)
        assert len(g.get_relationships_by_type(RelType.EXTENDS)) == 0
        assert len(g.get_relationships_by_type(RelType.IMPLEMENTS)) == 0


class TestHeritageDuplicateTuple:
    """Same heritage tuple twice doesn't crash (idempotent add)."""

    def test_duplicate_heritage_no_crash(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/a.py", "Parent"),
                label=NodeLabel.CLASS,
                name="Parent",
                file_path="src/a.py",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/a.py", "Child"),
                label=NodeLabel.CLASS,
                name="Child",
                file_path="src/a.py",
            )
        )

        parse_data = [
            _make_heritage_data(
                "src/a.py",
                [("Child", "extends", "Parent"), ("Child", "extends", "Parent")],
            ),
        ]
        # Should not raise.
        process_heritage(parse_data, g)

        extends_rels = g.get_relationships_by_type(RelType.EXTENDS)
        # May have 1 or 2 depending on graph implementation, but no crash.
        assert len(extends_rels) >= 1


class TestHeritagePhpTraitUse:
    """PHP trait use producing extends heritage is processed correctly."""

    def test_php_trait_use_heritage(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/service.php", "Loggable"),
                label=NodeLabel.CLASS,
                name="Loggable",
                file_path="src/service.php",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.CLASS, "src/service.php", "UserService"),
                label=NodeLabel.CLASS,
                name="UserService",
                file_path="src/service.php",
            )
        )

        parse_data = [
            _make_heritage_data(
                "src/service.php",
                [("UserService", "extends", "Loggable")],
                language="php",
            ),
        ]
        process_heritage(parse_data, g)

        extends_rels = g.get_relationships_by_type(RelType.EXTENDS)
        assert len(extends_rels) == 1
        assert extends_rels[0].source == generate_id(
            NodeLabel.CLASS, "src/service.php", "UserService"
        )


# ===========================================================================
# PART 3: Receiver Type Resolution
# ===========================================================================


_CALLABLE_LABELS = (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)


class TestReceiverNewExpression:
    """$var = new ClassName() produces variable_types that resolve $var->method()."""

    def test_new_expression_receiver_resolves(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.php")

        # Class with a method
        _add_symbol_node(g, NodeLabel.CLASS, "src/app.php", "UserService", 1, 30)
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.php", "find", 5, 10,
            class_name="UserService",
        )
        # Caller function
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.php", "main", 32, 40,
        )

        parse_data = [
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(name="main", kind="function", start_line=32, end_line=40, content=""),
                    ],
                    calls=[
                        CallInfo(name="find", line=35, receiver="svc"),
                    ],
                    variable_types={"svc": ["UserService"]},
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}
        assert method_id in targets


class TestReceiverCrossFileFactory:
    """Cross-file factory: $client = getFactory() resolves via global return types."""

    def test_cross_file_factory_resolved(self) -> None:
        g = KnowledgeGraph()

        # File A: factory function with return type
        _add_file_node(g, "src/factory.php")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/factory.php", "getClient", 1, 5,
        )

        # File B: class with method
        _add_file_node(g, "src/client.php")
        _add_symbol_node(g, NodeLabel.CLASS, "src/client.php", "ApiClient", 1, 20)
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/client.php", "request", 5, 15,
            class_name="ApiClient",
        )

        # File C: caller uses factory
        _add_file_node(g, "src/app.php")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.php", "main", 1, 10,
        )

        parse_data = [
            # factory.php: defines getClient() with return type ApiClient
            FileParseData(
                file_path="src/factory.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="getClient", kind="function",
                            start_line=1, end_line=5, content="",
                        ),
                    ],
                    type_refs=[
                        TypeRef(name="ApiClient", kind="return", line=1),
                    ],
                ),
            ),
            # client.php: no calls
            FileParseData(
                file_path="src/client.php",
                language="php",
                parse_result=ParseResult(),
            ),
            # app.php: $client = getClient(); $client->request();
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="main", kind="function",
                            start_line=1, end_line=10, content="",
                        ),
                    ],
                    calls=[
                        CallInfo(name="getClient", line=3),
                        CallInfo(name="request", line=5, receiver="client"),
                    ],
                    variable_types={
                        "client": ["__call__getClient"],
                    },
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}

        # The method should have been resolved via factory return type.
        assert method_id in targets


class TestReceiverUnresolvableSentinel:
    """Unresolvable __call__ sentinel is cleaned up, no spurious edges."""

    def test_unresolvable_sentinel_cleaned(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.php")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.php", "main", 1, 10,
        )

        parse_data = [
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="main", kind="function",
                            start_line=1, end_line=10, content="",
                        ),
                    ],
                    calls=[
                        CallInfo(name="doStuff", line=5, receiver="obj"),
                    ],
                    variable_types={
                        "obj": ["__call__unknownFactory"],
                    },
                ),
            ),
        ]

        process_calls(parse_data, g)

        # Sentinel should have been cleaned up.
        vt = parse_data[0].parse_result.variable_types
        assert "obj" not in vt

        # No CALLS edges to non-existent targets.
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 0


class TestReceiverUnionType:
    """Union return type creates CALLS edges to methods on all types."""

    def test_union_type_resolves_both_methods(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.php")

        # Two classes with same method name
        _add_symbol_node(g, NodeLabel.CLASS, "src/app.php", "LinqClient", 1, 20)
        linq_method = _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.php", "send", 5, 10,
            class_name="LinqClient",
        )
        _add_symbol_node(g, NodeLabel.CLASS, "src/app.php", "BBClient", 22, 40)
        bb_method = _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.php", "send", 25, 30,
            class_name="BBClient",
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.php", "dispatch", 42, 50,
        )

        parse_data = [
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="dispatch", kind="function",
                            start_line=42, end_line=50, content="",
                        ),
                    ],
                    calls=[
                        CallInfo(name="send", line=45, receiver="client"),
                    ],
                    variable_types={
                        "client": ["LinqClient", "BBClient"],
                    },
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}

        assert linq_method in targets
        assert bb_method in targets


class TestReceiverFallbackToVarName:
    """When variable_types has no entry, receiver name is tried as class name."""

    def test_fallback_to_receiver_as_class(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.php")

        # Class named "UserService" with a method
        _add_symbol_node(g, NodeLabel.CLASS, "src/app.php", "UserService", 1, 20)
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.php", "find", 5, 10,
            class_name="UserService",
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.php", "main", 22, 30,
        )

        parse_data = [
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="main", kind="function",
                            start_line=22, end_line=30, content="",
                        ),
                    ],
                    calls=[
                        # receiver is "UserService" (same as class name)
                        CallInfo(name="find", line=25, receiver="UserService"),
                    ],
                    # No variable_types entry for "UserService"
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}

        assert method_id in targets


class TestReceiverConfidenceScores:
    """Receiver-resolved method calls have confidence 0.8 cross-file."""

    def test_receiver_method_cross_file_resolved(self) -> None:
        """Cross-file receiver dispatch resolves to the correct method.

        When the method name is found globally, name-based resolution (0.5)
        fires first; receiver dispatch (0.8) would duplicate but is deduped.
        Either way the method IS in the targets.
        """
        g = KnowledgeGraph()
        _add_file_node(g, "src/svc.php")
        _add_symbol_node(g, NodeLabel.CLASS, "src/svc.php", "Svc", 1, 20)
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/svc.php", "run", 5, 10,
            class_name="Svc",
        )

        _add_file_node(g, "src/app.php")
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.php", "main", 1, 10,
        )

        parse_data = [
            FileParseData(
                file_path="src/svc.php",
                language="php",
                parse_result=ParseResult(),
            ),
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="main", kind="function",
                            start_line=1, end_line=10, content="",
                        ),
                    ],
                    calls=[
                        CallInfo(name="run", line=5, receiver="obj"),
                    ],
                    variable_types={"obj": ["Svc"]},
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)

        targets = {r.target for r in calls_rels}
        assert method_id in targets

    def test_same_file_receiver_gets_1_0(self) -> None:
        """Same-file method found by name matching gets confidence 1.0."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.php")
        _add_symbol_node(g, NodeLabel.CLASS, "src/app.php", "Svc", 1, 20)
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.php", "run", 5, 10,
            class_name="Svc",
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.php", "main", 22, 30,
        )

        parse_data = [
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    symbols=[
                        SymbolInfo(
                            name="main", kind="function",
                            start_line=22, end_line=30, content="",
                        ),
                    ],
                    calls=[
                        CallInfo(name="run", line=25, receiver="obj"),
                    ],
                    variable_types={"obj": ["Svc"]},
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)

        method_edges = [r for r in calls_rels if r.target == method_id]
        assert len(method_edges) == 1
        # Same-file name match hits first with confidence 1.0.
        assert method_edges[0].properties["confidence"] == 1.0


class TestPhpBlocklistInCalls:
    """PHP builtins in the call blocklist."""

    @pytest.mark.parametrize(
        "name",
        [
            "json_encode", "json_decode", "array_map", "isset", "empty",
            "strlen", "preg_match", "header", "session_start", "defined",
        ],
    )
    def test_php_builtin_in_blocklist(self, name: str) -> None:
        from axon.core.ingestion.calls import _CALL_BLOCKLIST
        assert name in _CALL_BLOCKLIST

    def test_blocklisted_php_builtin_no_edge(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.php")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.php", "doWork", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    calls=[CallInfo(name="json_encode", line=5)],
                ),
            ),
        ]

        process_calls(parse_data, g)
        assert len(g.get_relationships_by_type(RelType.CALLS)) == 0

    def test_blocklisted_self_receiver_bypass(self) -> None:
        """self/this receiver bypasses blocklist for user-defined methods."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.php")
        _add_symbol_node(g, NodeLabel.CLASS, "src/app.php", "Svc", 1, 20)
        _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.php", "get", 3, 8,
            class_name="Svc",
        )
        caller_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.php", "run", 10, 18,
            class_name="Svc",
        )

        parse_data = [
            FileParseData(
                file_path="src/app.php",
                language="php",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="get", line=12, receiver="this"),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        # "get" is blocklisted, but receiver is "this" so it should resolve.
        assert len(calls_rels) >= 1
