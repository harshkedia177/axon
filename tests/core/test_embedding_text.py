from __future__ import annotations

from axon.core.embeddings.text import generate_text, build_class_method_index, _MAX_TEXT_TOKENS
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)


def _node(
    label: NodeLabel,
    name: str,
    file_path: str = "",
    signature: str = "",
    class_name: str = "",
    **extra: object,
) -> GraphNode:
    """Build a ``GraphNode`` with a deterministic id."""
    return GraphNode(
        id=generate_id(label, file_path, name),
        label=label,
        name=name,
        file_path=file_path,
        signature=signature,
        class_name=class_name,
        **({"properties": extra} if extra else {}),
    )

def _rel(
    source: str,
    target: str,
    rel_type: RelType,
    rel_id: str | None = None,
) -> GraphRelationship:
    """Build a ``GraphRelationship``."""
    return GraphRelationship(
        id=rel_id or f"{rel_type.value}:{source}->{target}",
        type=rel_type,
        source=source,
        target=target,
    )

def _add(graph: KnowledgeGraph, *nodes: GraphNode) -> None:
    """Add multiple nodes to *graph*."""
    for n in nodes:
        graph.add_node(n)

class TestFunctionText:
    def test_function_basic_info(self) -> None:
        graph = KnowledgeGraph()
        fn = _node(
            NodeLabel.FUNCTION,
            "validate_user",
            file_path="src/auth.py",
            signature="def validate_user(user: User) -> bool",
        )
        graph.add_node(fn)

        text = generate_text(fn, graph)

        assert "function validate_user" in text
        assert "src/auth.py" in text
        assert "def validate_user(user: User) -> bool" in text

    def test_function_with_calls(self) -> None:
        graph = KnowledgeGraph()
        fn = _node(NodeLabel.FUNCTION, "validate_user", file_path="src/auth.py")
        callee1 = _node(NodeLabel.FUNCTION, "check_password", file_path="src/auth.py")
        callee2 = _node(NodeLabel.FUNCTION, "load_user", file_path="src/db.py")
        caller = _node(NodeLabel.FUNCTION, "login_handler", file_path="src/routes.py")
        _add(graph, fn, callee1, callee2, caller)

        # fn -> callee1, fn -> callee2 (outgoing CALLS)
        graph.add_relationship(_rel(fn.id, callee1.id, RelType.CALLS))
        graph.add_relationship(_rel(fn.id, callee2.id, RelType.CALLS))
        # caller -> fn (incoming CALLS)
        graph.add_relationship(_rel(caller.id, fn.id, RelType.CALLS))

        text = generate_text(fn, graph)

        assert "calls:" in text.lower() or "calls:" in text
        assert "check_password" in text
        assert "load_user" in text
        assert "called by:" in text.lower() or "called by:" in text
        assert "login_handler" in text

    def test_function_with_uses_type(self) -> None:
        graph = KnowledgeGraph()
        fn = _node(NodeLabel.FUNCTION, "validate_user", file_path="src/auth.py")
        type_node = _node(NodeLabel.CLASS, "User", file_path="src/models.py")
        _add(graph, fn, type_node)

        graph.add_relationship(_rel(fn.id, type_node.id, RelType.USES_TYPE))

        text = generate_text(fn, graph)

        assert "uses types:" in text.lower() or "uses types:" in text
        assert "User" in text

class TestMethodText:
    def test_method_includes_class_name(self) -> None:
        graph = KnowledgeGraph()
        method = _node(
            NodeLabel.METHOD,
            "get_name",
            file_path="src/models.py",
            signature="def get_name(self) -> str",
            class_name="User",
        )
        graph.add_node(method)

        text = generate_text(method, graph)

        assert "method get_name" in text
        assert "src/models.py" in text
        assert "User" in text
        assert "def get_name(self) -> str" in text

class TestClassText:
    def test_class_basic_info(self) -> None:
        graph = KnowledgeGraph()
        cls = _node(NodeLabel.CLASS, "UserService", file_path="src/services.py")
        graph.add_node(cls)

        text = generate_text(cls, graph)

        assert "class UserService" in text
        assert "src/services.py" in text

    def test_class_with_methods(self) -> None:
        graph = KnowledgeGraph()
        cls = _node(NodeLabel.CLASS, "UserService", file_path="src/services.py")
        m1 = _node(
            NodeLabel.METHOD,
            "create_user",
            file_path="src/services.py",
            class_name="UserService",
        )
        m2 = _node(
            NodeLabel.METHOD,
            "delete_user",
            file_path="src/services.py",
            class_name="UserService",
        )
        # Unrelated method in same file, different class
        m_other = _node(
            NodeLabel.METHOD,
            "other_method",
            file_path="src/services.py",
            class_name="OtherClass",
        )
        _add(graph, cls, m1, m2, m_other)

        text = generate_text(cls, graph)

        assert "create_user" in text
        assert "delete_user" in text
        assert "other_method" not in text

    def test_class_with_extends_and_implements(self) -> None:
        graph = KnowledgeGraph()
        cls = _node(NodeLabel.CLASS, "Admin", file_path="src/models.py")
        base = _node(NodeLabel.CLASS, "User", file_path="src/models.py")
        iface = _node(NodeLabel.INTERFACE, "Serializable", file_path="src/types.py")
        _add(graph, cls, base, iface)

        graph.add_relationship(_rel(cls.id, base.id, RelType.EXTENDS))
        graph.add_relationship(_rel(cls.id, iface.id, RelType.IMPLEMENTS))

        text = generate_text(cls, graph)

        assert "extends:" in text.lower() or "extends:" in text
        assert "User" in text
        assert "implements:" in text.lower() or "implements:" in text
        assert "Serializable" in text

class TestFileText:
    def test_file_basic_info(self) -> None:
        graph = KnowledgeGraph()
        file_node = _node(NodeLabel.FILE, "auth.py", file_path="src/auth.py")
        graph.add_node(file_node)

        text = generate_text(file_node, graph)

        assert "file auth.py" in text
        assert "src/auth.py" in text

    def test_file_with_defines_and_imports(self) -> None:
        graph = KnowledgeGraph()
        file_node = _node(NodeLabel.FILE, "auth.py", file_path="src/auth.py")
        fn = _node(NodeLabel.FUNCTION, "validate", file_path="src/auth.py")
        cls = _node(NodeLabel.CLASS, "AuthService", file_path="src/auth.py")
        imported = _node(NodeLabel.FUNCTION, "hash_password", file_path="src/crypto.py")
        _add(graph, file_node, fn, cls, imported)

        graph.add_relationship(_rel(file_node.id, fn.id, RelType.DEFINES))
        graph.add_relationship(_rel(file_node.id, cls.id, RelType.DEFINES))
        graph.add_relationship(_rel(file_node.id, imported.id, RelType.IMPORTS))

        text = generate_text(file_node, graph)

        assert "defines:" in text.lower() or "defines:" in text
        assert "validate" in text
        assert "AuthService" in text
        assert "imports:" in text.lower() or "imports:" in text
        assert "hash_password" in text

class TestInterfaceText:
    def test_interface_basic(self) -> None:
        graph = KnowledgeGraph()
        iface = _node(
            NodeLabel.INTERFACE,
            "Serializable",
            file_path="src/types.ts",
            signature="interface Serializable { toJSON(): string; }",
        )
        graph.add_node(iface)

        text = generate_text(iface, graph)

        assert "interface Serializable" in text
        assert "src/types.ts" in text
        assert "interface Serializable { toJSON(): string; }" in text

class TestTypeAliasText:
    def test_type_alias_basic(self) -> None:
        graph = KnowledgeGraph()
        ta = _node(
            NodeLabel.TYPE_ALIAS,
            "UserID",
            file_path="src/types.py",
            signature="type UserID = int",
        )
        graph.add_node(ta)

        text = generate_text(ta, graph)

        assert "type_alias UserID" in text
        assert "src/types.py" in text
        assert "type UserID = int" in text

class TestEnumText:
    def test_enum_basic(self) -> None:
        graph = KnowledgeGraph()
        enum_node = _node(
            NodeLabel.ENUM,
            "Color",
            file_path="src/enums.py",
            signature="class Color(Enum): RED = 1; GREEN = 2; BLUE = 3",
        )
        graph.add_node(enum_node)

        text = generate_text(enum_node, graph)

        assert "enum Color" in text
        assert "src/enums.py" in text
        assert "class Color(Enum)" in text

class TestFolderText:
    def test_folder_with_contents(self) -> None:
        graph = KnowledgeGraph()
        folder = _node(NodeLabel.FOLDER, "auth", file_path="src/auth")
        f1 = _node(NodeLabel.FILE, "validate.py", file_path="src/auth/validate.py")
        f2 = _node(NodeLabel.FILE, "hash.py", file_path="src/auth/hash.py")
        _add(graph, folder, f1, f2)

        graph.add_relationship(_rel(folder.id, f1.id, RelType.CONTAINS))
        graph.add_relationship(_rel(folder.id, f2.id, RelType.CONTAINS))

        text = generate_text(folder, graph)

        assert "folder auth" in text
        assert "src/auth" in text
        assert "contains:" in text.lower() or "contains:" in text
        assert "validate.py" in text
        assert "hash.py" in text

class TestCommunityText:
    def test_community_with_members(self) -> None:
        graph = KnowledgeGraph()
        community = _node(NodeLabel.COMMUNITY, "Auth")
        member1 = _node(NodeLabel.FUNCTION, "validate", file_path="src/auth.py")
        member2 = _node(NodeLabel.FUNCTION, "hash_password", file_path="src/auth.py")
        _add(graph, community, member1, member2)

        # MEMBER_OF: member -> community (incoming to community)
        graph.add_relationship(_rel(member1.id, community.id, RelType.MEMBER_OF))
        graph.add_relationship(_rel(member2.id, community.id, RelType.MEMBER_OF))

        text = generate_text(community, graph)

        assert "community Auth" in text
        assert "members:" in text.lower() or "members:" in text
        assert "validate" in text
        assert "hash_password" in text

class TestProcessText:
    def test_process_with_steps(self) -> None:
        graph = KnowledgeGraph()
        process = _node(NodeLabel.PROCESS, "user_registration")
        step1 = _node(NodeLabel.FUNCTION, "validate_input", file_path="src/reg.py")
        step2 = _node(NodeLabel.FUNCTION, "create_user", file_path="src/reg.py")
        step3 = _node(NodeLabel.FUNCTION, "send_email", file_path="src/email.py")
        _add(graph, process, step1, step2, step3)

        # STEP_IN_PROCESS: step -> process (incoming to process)
        graph.add_relationship(_rel(step1.id, process.id, RelType.STEP_IN_PROCESS))
        graph.add_relationship(_rel(step2.id, process.id, RelType.STEP_IN_PROCESS))
        graph.add_relationship(_rel(step3.id, process.id, RelType.STEP_IN_PROCESS))

        text = generate_text(process, graph)

        assert "process user_registration" in text
        assert "steps:" in text.lower() or "steps:" in text
        assert "validate_input" in text
        assert "create_user" in text
        assert "send_email" in text

class TestEdgeCases:
    def test_node_with_no_edges(self) -> None:
        graph = KnowledgeGraph()
        fn = _node(
            NodeLabel.FUNCTION,
            "orphan_func",
            file_path="src/utils.py",
            signature="def orphan_func() -> None",
        )
        graph.add_node(fn)

        text = generate_text(fn, graph)

        assert "function orphan_func" in text
        assert "src/utils.py" in text
        # Should not contain empty "calls:" or "called by:" sections
        assert text.strip()  # Non-empty

    def test_empty_signature_is_omitted(self) -> None:
        graph = KnowledgeGraph()
        fn = _node(NodeLabel.FUNCTION, "simple", file_path="src/app.py", signature="")
        graph.add_node(fn)

        text = generate_text(fn, graph)

        assert "signature:" not in text.lower()

    def test_method_with_calls_and_types(self) -> None:
        graph = KnowledgeGraph()
        method = _node(
            NodeLabel.METHOD,
            "process",
            file_path="src/service.py",
            signature="def process(self, data: Data) -> Result",
            class_name="Worker",
        )
        callee = _node(NodeLabel.FUNCTION, "transform", file_path="src/utils.py")
        type_node = _node(NodeLabel.CLASS, "Data", file_path="src/models.py")
        _add(graph, method, callee, type_node)

        graph.add_relationship(_rel(method.id, callee.id, RelType.CALLS))
        graph.add_relationship(_rel(method.id, type_node.id, RelType.USES_TYPE))

        text = generate_text(method, graph)

        assert "method process" in text
        assert "Worker" in text
        assert "transform" in text
        assert "Data" in text


def _make_hub_function(num_callees: int = 100) -> tuple[KnowledgeGraph, GraphNode]:
    """Create a function node that calls many other functions."""
    graph = KnowledgeGraph()
    hub = GraphNode(
        id="function:src/hub.py:hub_func",
        label=NodeLabel.FUNCTION,
        name="hub_func",
        file_path="src/hub.py",
        signature="def hub_func(a, b, c, d, e):",
    )
    graph.add_node(hub)
    for i in range(num_callees):
        callee = GraphNode(
            id=f"function:src/mod{i}.py:callee_{i}",
            label=NodeLabel.FUNCTION,
            name=f"callee_with_a_long_name_{i}",
            file_path=f"src/mod{i}.py",
        )
        graph.add_node(callee)
        graph.add_relationship(GraphRelationship(
            id=f"calls:{hub.id}->{callee.id}",
            type=RelType.CALLS,
            source=hub.id,
            target=callee.id,
        ))
    return graph, hub


class TestTextBudget:
    def test_hub_function_stays_within_budget(self) -> None:
        graph, hub = _make_hub_function(num_callees=200)
        idx = build_class_method_index(graph)
        text = generate_text(hub, graph, idx)
        max_chars = _MAX_TEXT_TOKENS * 4
        assert len(text) <= max_chars, f"Text is {len(text)} chars, budget is {max_chars}"

    def test_small_function_is_not_truncated(self) -> None:
        graph = KnowledgeGraph()
        fn = GraphNode(
            id="function:src/a.py:small",
            label=NodeLabel.FUNCTION,
            name="small",
            file_path="src/a.py",
            signature="def small():",
        )
        graph.add_node(fn)
        idx = build_class_method_index(graph)
        text = generate_text(fn, graph, idx)
        assert "small" in text
        assert "signature:" in text
