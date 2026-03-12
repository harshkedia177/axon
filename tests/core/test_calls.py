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
from axon.core.ingestion.calls import (
    _CALL_BLOCKLIST,
    process_calls,
    resolve_call,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import build_name_index
from axon.core.parsers.base import CallInfo, FuncRef, ParseResult, TypeRef

_CALLABLE_LABELS = (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)


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


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the test fixture specification.

    File: src/auth.py
        Function: validate (lines 1-10)
        Function: hash_password (lines 12-20)

    File: src/app.py
        Function: login (lines 1-15)

    File: src/utils.py
        Function: helper (lines 1-5)
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/app.py")
    _add_file_node(g, "src/utils.py")

    # Symbols in src/auth.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10)
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "hash_password", 12, 20)

    # Symbols in src/app.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "login", 1, 15)

    # Symbols in src/utils.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/utils.py", "helper", 1, 5)

    return g


@pytest.fixture()
def parse_data() -> list[FileParseData]:
    """Parse data with calls matching the fixture specification.

    src/auth.py: hash_password() at line 5 (inside validate)
    src/app.py: validate() at line 8 (inside login)
    """
    return [
        FileParseData(
            file_path="src/auth.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="hash_password", line=5)],
            ),
        ),
        FileParseData(
            file_path="src/app.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="validate", line=8)],
            ),
        ),
    ]


class TestBuildCallIndex:
    def test_build_call_index(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)

        # All four functions should appear.
        assert "validate" in index
        assert "hash_password" in index
        assert "login" in index
        assert "helper" in index

        # Each name maps to exactly one node ID.
        assert len(index["validate"]) == 1
        assert len(index["hash_password"]) == 1

        # IDs match expected generate_id output.
        expected_validate = generate_id(NodeLabel.FUNCTION, "src/auth.py", "validate")
        assert index["validate"] == [expected_validate]

    def test_build_call_index_includes_classes(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "User", 1, 20)

        index = build_name_index(g, _CALLABLE_LABELS)
        assert "User" in index
        assert len(index["User"]) == 1

    def test_build_call_index_multiple_same_name(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/a.py")
        _add_file_node(g, "src/b.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/a.py", "init", 1, 5)
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/b.py", "init", 1, 5)

        index = build_name_index(g, _CALLABLE_LABELS)
        assert "init" in index
        assert len(index["init"]) == 2


class TestResolveCallSameFile:
    def test_resolve_call_same_file(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="hash_password", line=5)

        target_id, confidence = resolve_call(call, "src/auth.py", index, graph)

        expected_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "hash_password")
        assert target_id == expected_id
        assert confidence == 1.0


class TestResolveCallGlobal:
    def test_resolve_call_global(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="validate", line=8)

        target_id, confidence = resolve_call(call, "src/app.py", index, graph)

        expected_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "validate")
        assert target_id == expected_id
        assert confidence == 0.5


class TestResolveCallUnresolved:
    def test_resolve_call_unresolved(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="nonexistent_function", line=3)

        target_id, confidence = resolve_call(call, "src/auth.py", index, graph)

        assert target_id is None
        assert confidence == 0.0


class TestProcessCallsCreatesRelationships:
    def test_process_calls_creates_relationships(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_calls(parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 2

        # Collect source->target pairs.
        pairs = {(r.source, r.target) for r in calls_rels}

        validate_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "validate")
        hash_pw_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "hash_password")
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        # validate -> hash_password (same-file call at line 5 inside validate)
        assert (validate_id, hash_pw_id) in pairs
        # login -> validate (cross-file call at line 8 inside login)
        assert (login_id, validate_id) in pairs


class TestProcessCallsConfidence:
    def test_process_calls_confidence(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_calls(parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)

        validate_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "validate")
        hash_pw_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "hash_password")
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        confidences = {(r.source, r.target): r.properties["confidence"] for r in calls_rels}

        # Same-file call: confidence 1.0
        assert confidences[(validate_id, hash_pw_id)] == 1.0
        # Cross-file global match: confidence 0.5
        assert confidences[(login_id, validate_id)] == 0.5


class TestProcessCallsNoDuplicates:
    def test_process_calls_no_duplicates(self, graph: KnowledgeGraph) -> None:
        # Two identical calls to hash_password inside validate.
        duplicate_parse_data = [
            FileParseData(
                file_path="src/auth.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="hash_password", line=5),
                        CallInfo(name="hash_password", line=7),
                    ],
                ),
            ),
        ]

        process_calls(duplicate_parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        # Both calls resolve to validate -> hash_password, but only one
        # relationship should exist.
        assert len(calls_rels) == 1


class TestResolveMethodCallSelf:
    def test_resolve_method_call_self(self) -> None:
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.py")
        _add_symbol_node(
            g,
            NodeLabel.CLASS,
            "src/service.py",
            "AuthService",
            1,
            30,
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.py",
            "login",
            3,
            15,
            class_name="AuthService",
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.py",
            "check_token",
            17,
            28,
            class_name="AuthService",
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="check_token", line=10, receiver="self")

        target_id, confidence = resolve_call(call, "src/service.py", index, g)

        expected_id = generate_id(NodeLabel.METHOD, "src/service.py", "AuthService.check_token")
        assert target_id == expected_id
        assert confidence == 1.0

    def test_resolve_method_call_this(self) -> None:
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.ts")
        _add_symbol_node(
            g,
            NodeLabel.CLASS,
            "src/service.ts",
            "AuthService",
            1,
            30,
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.ts",
            "checkToken",
            17,
            28,
            class_name="AuthService",
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="checkToken", line=10, receiver="this")

        target_id, confidence = resolve_call(call, "src/service.ts", index, g)

        expected_id = generate_id(NodeLabel.METHOD, "src/service.ts", "AuthService.checkToken")
        assert target_id == expected_id
        assert confidence == 1.0


class TestResolveCallImportResolved:
    def test_resolve_call_import_resolved(self) -> None:
        g = KnowledgeGraph()

        # Two files: app.py imports validate from auth.py.
        _add_file_node(g, "src/auth.py")
        _add_file_node(g, "src/app.py")

        _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10)
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "login", 1, 15)

        # IMPORTS relationship: app.py -> auth.py with symbol "validate"
        app_file_id = generate_id(NodeLabel.FILE, "src/app.py")
        auth_file_id = generate_id(NodeLabel.FILE, "src/auth.py")
        g.add_relationship(
            GraphRelationship(
                id=f"imports:{app_file_id}->{auth_file_id}",
                type=RelType.IMPORTS,
                source=app_file_id,
                target=auth_file_id,
                properties={"symbols": "validate"},
            )
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="validate", line=8)

        target_id, confidence = resolve_call(call, "src/app.py", index, g)

        expected_id = generate_id(NodeLabel.FUNCTION, "src/auth.py", "validate")
        assert target_id == expected_id
        assert confidence == 1.0


class TestCallBlocklist:
    def test_blocklist_is_frozenset(self) -> None:
        assert isinstance(_CALL_BLOCKLIST, frozenset)

    def test_python_builtins_in_blocklist(self) -> None:
        for name in ("print", "len", "range", "isinstance", "super"):
            assert name in _CALL_BLOCKLIST

    def test_js_globals_in_blocklist(self) -> None:
        for name in ("console", "setTimeout", "fetch", "JSON", "Promise"):
            assert name in _CALL_BLOCKLIST

    def test_react_hooks_in_blocklist(self) -> None:
        for name in ("useState", "useEffect", "useCallback", "useMemo"):
            assert name in _CALL_BLOCKLIST

    def test_blocklisted_call_creates_no_edge(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "do_work", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="print", line=5)],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 0

    def test_blocklisted_argument_creates_no_edge(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "do_work", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="apply_func", line=5, arguments=["str"]),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        # apply_func is not in the graph so no edge for it; 'str' is blocklisted.
        assert len(calls_rels) == 0

    def test_non_blocklisted_call_still_resolves(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "caller", 1, 10)
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "my_helper", 12, 20)

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="my_helper", line=5)],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 1


class TestModuleLevelCall:
    """Module-level calls should be attributed to the File node."""

    def test_module_level_call_creates_edge_from_file_node(self) -> None:
        g = KnowledgeGraph()
        file_id = _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "setup", 10, 20)

        # Call at line 1 is outside any function (module-level).
        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="setup", line=1)],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 1
        assert calls_rels[0].source == file_id

    def test_module_level_call_no_file_node_skipped(self) -> None:
        g = KnowledgeGraph()
        # No file node, only a function.
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "setup", 10, 20)

        parse_data = [
            FileParseData(
                file_path="src/orphan.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="setup", line=1)],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 0


class TestTypeInferenceConstructor:
    """Type inference: ``user = User()`` then ``user.save()`` resolves to User.save."""

    def test_constructor_type_inference(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.py")
        _add_file_node(g, "src/models.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "User", 1, 20)
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "save",
            10,
            15,
            class_name="User",
        )
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "main", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/app.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="User", line=3, assignment_target="user"),
                        CallInfo(name="save", line=5, receiver="user"),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)

        user_save_id = generate_id(NodeLabel.METHOD, "src/models.py", "User.save")
        targets = {r.target for r in calls_rels}
        assert user_save_id in targets


class TestTypeInferenceAnnotation:
    """Type inference: ``user: User = get_user()`` then ``user.validate()`` resolves."""

    def test_annotation_type_inference(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.py")
        _add_file_node(g, "src/models.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "User", 1, 20)
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "validate",
            10,
            15,
            class_name="User",
        )
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "main", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/app.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="validate", line=5, receiver="user"),
                    ],
                    type_refs=[
                        TypeRef(name="User", kind="variable", line=3, variable_name="user"),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)

        user_validate_id = generate_id(NodeLabel.METHOD, "src/models.py", "User.validate")
        targets = {r.target for r in calls_rels}
        assert user_validate_id in targets


class TestTypeInferenceParam:
    """Type inference: ``def process(user: User)`` then ``user.save()`` resolves."""

    def test_param_type_inference(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.py")
        _add_file_node(g, "src/models.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "User", 1, 20)
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "save",
            10,
            15,
            class_name="User",
        )
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "process", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/app.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="save", line=5, receiver="user"),
                    ],
                    type_refs=[
                        TypeRef(name="User", kind="param", line=1, param_name="user"),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)

        user_save_id = generate_id(NodeLabel.METHOD, "src/models.py", "User.save")
        targets = {r.target for r in calls_rels}
        assert user_save_id in targets


class TestTypeInferenceNoMatchFallback:
    """Without type info, receiver method resolution falls back to literal comparison."""

    def test_no_type_info_no_receiver_match(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/app.py")
        _add_file_node(g, "src/other.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/other.py", "User", 1, 20)
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/other.py",
            "save",
            10,
            15,
            class_name="User",
        )
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "main", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/app.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="save", line=5, receiver="user"),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)

        # Without type info, "user" != "User" so receiver method resolution fails.
        # The call may still resolve via global fuzzy match but NOT at 0.8 confidence.
        user_save_id = generate_id(NodeLabel.METHOD, "src/other.py", "User.save")
        for rel in calls_rels:
            if rel.target == user_save_id:
                assert rel.properties["confidence"] != 0.8


class TestFuncRefAssignment:
    """First-class function references should create CALLS edges."""

    def test_func_ref_creates_edge(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "my_func", 1, 10)
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "setup", 12, 20)

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    func_refs=[FuncRef(name="my_func", line=15, target_var="handler")],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)

        my_func_id = generate_id(NodeLabel.FUNCTION, "src/main.py", "my_func")
        targets = {r.target for r in calls_rels}
        assert my_func_id in targets

        # Confidence should be 0.7 of same-file match (1.0 * 0.7 = 0.7)
        for rel in calls_rels:
            if rel.target == my_func_id:
                assert rel.properties["confidence"] == pytest.approx(0.7)

    def test_func_ref_module_level(self) -> None:
        g = KnowledgeGraph()
        file_id = _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "my_func", 10, 20)

        # FuncRef at line 1 is outside any function.
        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    func_refs=[FuncRef(name="my_func", line=1, target_var="handler")],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 1
        assert calls_rels[0].source == file_id

    def test_blocklisted_func_ref_creates_no_edge(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "setup", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    func_refs=[FuncRef(name="print", line=5, target_var="handler")],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 0
