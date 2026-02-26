"""Dead code predicate unit tests.

Covers the individual exemption predicates that are not exercised by
the integration-level test_dead_code.py:

- _is_test_file (various conventions)
- _is_example_file
- _is_html_file
- _is_framework_method (migrations, seeders, commands, etc.)
- _is_test_method / _is_test_class
- _is_enum_class
- _is_python_public_api
- _has_property_decorator
- _has_typing_stub_decorator
- _is_subclassed
- _clear_override_false_positives
- _clear_protocol_stub_false_positives
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
    _is_test_file,
    _is_example_file,
    _is_html_file,
    _is_framework_method,
    _is_test_method,
    _is_test_class,
    _is_enum_class,
    _is_python_public_api,
    _has_property_decorator,
    _has_typing_stub_decorator,
    _is_subclassed,
    _is_dunder,
    _is_exempt,
    process_dead_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    label: NodeLabel = NodeLabel.FUNCTION,
    file_path: str = "src/app.py",
    name: str = "func",
    *,
    class_name: str = "",
    properties: dict | None = None,
) -> GraphNode:
    symbol_name = (
        f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
    )
    node_id = generate_id(label, file_path, symbol_name)
    return GraphNode(
        id=node_id,
        label=label,
        name=name,
        file_path=file_path,
        class_name=class_name,
        properties=properties or {},
    )


def _add_symbol(
    graph: KnowledgeGraph,
    label: NodeLabel,
    file_path: str,
    name: str,
    *,
    class_name: str = "",
    is_entry_point: bool = False,
    is_exported: bool = False,
    properties: dict | None = None,
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
            class_name=class_name,
            is_entry_point=is_entry_point,
            is_exported=is_exported,
            properties=properties or {},
        )
    )
    return node_id


# ---------------------------------------------------------------------------
# _is_test_file
# ---------------------------------------------------------------------------


class TestIsTestFile:
    """_is_test_file identifies test files across Python, JS/TS, and PHP."""

    def test_python_tests_dir(self) -> None:
        assert _is_test_file("src/tests/test_auth.py") is True

    def test_python_tests_prefix(self) -> None:
        assert _is_test_file("tests/unit/test_foo.py") is True

    def test_python_conftest(self) -> None:
        assert _is_test_file("tests/conftest.py") is True

    def test_python_test_underscore_in_path(self) -> None:
        # /test_ in path matches (test_helpers.py is a test file)
        assert _is_test_file("src/test_helpers.py") is True
        assert _is_test_file("src/utils/test_data.py") is True

    def test_js_test_dot(self) -> None:
        assert _is_test_file("src/components/Button.test.tsx") is True

    def test_js_spec_dot(self) -> None:
        assert _is_test_file("src/components/Button.spec.ts") is True

    def test_js_dunder_tests_dir(self) -> None:
        assert _is_test_file("src/__tests__/Button.tsx") is True

    def test_php_test_suffix(self) -> None:
        assert _is_test_file("tests/Unit/AuthTest.php") is True

    def test_php_tests_dir_uppercase(self) -> None:
        assert _is_test_file("Tests/Feature/ApiTest.php") is True

    def test_regular_file_not_test(self) -> None:
        assert _is_test_file("src/auth/login.py") is False

    def test_test_in_path_segment(self) -> None:
        # /test/ (singular) also matches
        assert _is_test_file("src/test/unit/foo.py") is True

    def test_tests_prefix_relative(self) -> None:
        assert _is_test_file("tests/foo.py") is True
        assert _is_test_file("Tests/foo.php") is True


# ---------------------------------------------------------------------------
# _is_example_file
# ---------------------------------------------------------------------------


class TestIsExampleFile:
    """_is_example_file identifies example/template/sample files."""

    def test_example_extension(self) -> None:
        assert _is_example_file("config/settings.example.php") is True

    def test_sample_extension(self) -> None:
        assert _is_example_file("config.sample.py") is True

    def test_template_extension(self) -> None:
        assert _is_example_file(".env.template.local") is True

    def test_examples_directory(self) -> None:
        assert _is_example_file("docs/examples/usage.py") is True

    def test_example_directory(self) -> None:
        assert _is_example_file("src/example/demo.ts") is True

    def test_regular_file_not_example(self) -> None:
        assert _is_example_file("src/config/settings.py") is False


# ---------------------------------------------------------------------------
# _is_html_file
# ---------------------------------------------------------------------------


class TestIsHtmlFile:
    """_is_html_file matches .html and .htm extensions."""

    def test_html_extension(self) -> None:
        assert _is_html_file("public/index.html") is True

    def test_htm_extension(self) -> None:
        assert _is_html_file("templates/page.htm") is True

    def test_php_not_html(self) -> None:
        assert _is_html_file("views/page.php") is False

    def test_js_not_html(self) -> None:
        assert _is_html_file("src/app.js") is False


# ---------------------------------------------------------------------------
# _is_framework_method
# ---------------------------------------------------------------------------


class TestIsFrameworkMethod:
    """_is_framework_method matches framework-invoked methods by directory."""

    def test_migration_up(self) -> None:
        assert _is_framework_method("up", "database/migrations/2024_create_users.php") is True

    def test_migration_down(self) -> None:
        assert _is_framework_method("down", "database/migrations/2024_create_users.php") is True

    def test_seeder_run(self) -> None:
        assert _is_framework_method("run", "database/seeders/UserSeeder.php") is True

    def test_seeds_run(self) -> None:
        assert _is_framework_method("run", "database/seeds/UserSeeder.php") is True

    def test_command_handle(self) -> None:
        assert _is_framework_method("handle", "app/Commands/SyncCommand.php") is True

    def test_listener_handle(self) -> None:
        assert _is_framework_method("handle", "app/Listeners/OrderShipped.php") is True

    def test_provider_register(self) -> None:
        assert _is_framework_method("register", "app/Providers/AppServiceProvider.php") is True

    def test_provider_boot(self) -> None:
        assert _is_framework_method("boot", "app/Providers/AppServiceProvider.php") is True

    def test_middleware_handle(self) -> None:
        assert _is_framework_method("handle", "app/Middleware/Auth.php") is True

    def test_job_handle(self) -> None:
        assert _is_framework_method("handle", "app/Jobs/SendEmail.php") is True

    def test_non_framework_method(self) -> None:
        assert _is_framework_method("handle", "src/utils/helper.py") is False

    def test_non_framework_name(self) -> None:
        assert _is_framework_method("doStuff", "database/migrations/2024_create.php") is False

    def test_lowercase_middleware(self) -> None:
        assert _is_framework_method("handle", "app/middleware/cors.ts") is True

    def test_lowercase_listeners(self) -> None:
        assert _is_framework_method("handle", "app/listeners/on_event.py") is True


# ---------------------------------------------------------------------------
# _is_test_method / _is_test_class
# ---------------------------------------------------------------------------


class TestIsTestMethod:
    """_is_test_method matches PHPUnit/JUnit camelCase test methods."""

    def test_phpunit_style(self) -> None:
        assert _is_test_method("testSanitizeEscapesHtml") is True

    def test_junit_style(self) -> None:
        assert _is_test_method("testEncryptRoundTrip") is True

    def test_python_style_not_matched(self) -> None:
        # Python test_ style is handled by _is_exempt directly
        assert _is_test_method("test_validate") is False

    def test_short_name_not_matched(self) -> None:
        assert _is_test_method("test") is False

    def test_lowercase_after_test_not_matched(self) -> None:
        assert _is_test_method("testing") is False


class TestIsTestClass:
    """_is_test_class matches pytest-style Test* classes."""

    def test_pytest_class(self) -> None:
        assert _is_test_class("TestHandleQuery") is True

    def test_short_name(self) -> None:
        assert _is_test_class("Test") is False

    def test_lowercase_after_test(self) -> None:
        assert _is_test_class("Tester") is False


# ---------------------------------------------------------------------------
# _is_enum_class
# ---------------------------------------------------------------------------


class TestIsEnumClass:
    """_is_enum_class matches classes with Enum bases."""

    def test_enum_base(self) -> None:
        node = _make_node(NodeLabel.CLASS, name="Color", properties={"bases": ["Enum"]})
        assert _is_enum_class(node, NodeLabel.CLASS) is True

    def test_int_enum_base(self) -> None:
        node = _make_node(NodeLabel.CLASS, name="Priority", properties={"bases": ["IntEnum"]})
        assert _is_enum_class(node, NodeLabel.CLASS) is True

    def test_str_enum_base(self) -> None:
        node = _make_node(NodeLabel.CLASS, name="Status", properties={"bases": ["StrEnum"]})
        assert _is_enum_class(node, NodeLabel.CLASS) is True

    def test_flag_base(self) -> None:
        node = _make_node(NodeLabel.CLASS, name="Perms", properties={"bases": ["Flag"]})
        assert _is_enum_class(node, NodeLabel.CLASS) is True

    def test_int_flag_base(self) -> None:
        node = _make_node(NodeLabel.CLASS, name="Bits", properties={"bases": ["IntFlag"]})
        assert _is_enum_class(node, NodeLabel.CLASS) is True

    def test_regular_class_no_enum(self) -> None:
        node = _make_node(NodeLabel.CLASS, name="User", properties={"bases": ["Model"]})
        assert _is_enum_class(node, NodeLabel.CLASS) is False

    def test_function_not_enum(self) -> None:
        node = _make_node(NodeLabel.FUNCTION, name="Color", properties={"bases": ["Enum"]})
        assert _is_enum_class(node, NodeLabel.FUNCTION) is False

    def test_no_bases_not_enum(self) -> None:
        node = _make_node(NodeLabel.CLASS, name="Color", properties={})
        assert _is_enum_class(node, NodeLabel.CLASS) is False


# ---------------------------------------------------------------------------
# _is_python_public_api
# ---------------------------------------------------------------------------


class TestIsPythonPublicApi:
    """_is_python_public_api matches public symbols in __init__.py."""

    def test_public_in_init(self) -> None:
        assert _is_python_public_api("create_app", "src/mypackage/__init__.py") is True

    def test_private_in_init(self) -> None:
        assert _is_python_public_api("_internal", "src/mypackage/__init__.py") is False

    def test_public_not_in_init(self) -> None:
        assert _is_python_public_api("create_app", "src/mypackage/main.py") is False


# ---------------------------------------------------------------------------
# _has_property_decorator
# ---------------------------------------------------------------------------


class TestHasPropertyDecorator:
    """_has_property_decorator matches @property decorated methods."""

    def test_property_decorator(self) -> None:
        node = _make_node(properties={"decorators": ["property"]})
        assert _has_property_decorator(node) is True

    def test_no_property(self) -> None:
        node = _make_node(properties={"decorators": ["staticmethod"]})
        assert _has_property_decorator(node) is False

    def test_empty_decorators(self) -> None:
        node = _make_node(properties={"decorators": []})
        assert _has_property_decorator(node) is False

    def test_no_decorators_key(self) -> None:
        node = _make_node(properties={})
        assert _has_property_decorator(node) is False


# ---------------------------------------------------------------------------
# _has_typing_stub_decorator
# ---------------------------------------------------------------------------


class TestHasTypingStubDecorator:
    """_has_typing_stub_decorator matches @overload and @abstractmethod."""

    def test_overload(self) -> None:
        node = _make_node(properties={"decorators": ["overload"]})
        assert _has_typing_stub_decorator(node) is True

    def test_typing_overload(self) -> None:
        node = _make_node(properties={"decorators": ["typing.overload"]})
        assert _has_typing_stub_decorator(node) is True

    def test_abstractmethod(self) -> None:
        node = _make_node(properties={"decorators": ["abstractmethod"]})
        assert _has_typing_stub_decorator(node) is True

    def test_abc_abstractmethod(self) -> None:
        node = _make_node(properties={"decorators": ["abc.abstractmethod"]})
        assert _has_typing_stub_decorator(node) is True

    def test_not_a_stub(self) -> None:
        node = _make_node(properties={"decorators": ["staticmethod"]})
        assert _has_typing_stub_decorator(node) is False


# ---------------------------------------------------------------------------
# _is_subclassed
# ---------------------------------------------------------------------------


class TestIsSubclassed:
    """_is_subclassed checks for incoming EXTENDS/IMPLEMENTS edges."""

    def test_class_with_extends(self) -> None:
        g = KnowledgeGraph()
        parent_id = _add_symbol(g, NodeLabel.CLASS, "src/base.py", "Base")
        child_id = _add_symbol(g, NodeLabel.CLASS, "src/child.py", "Child")
        g.add_relationship(
            GraphRelationship(
                id="ext:1",
                type=RelType.EXTENDS,
                source=child_id,
                target=parent_id,
            )
        )
        assert _is_subclassed(g, parent_id, NodeLabel.CLASS) is True

    def test_class_with_implements(self) -> None:
        g = KnowledgeGraph()
        iface_id = _add_symbol(g, NodeLabel.CLASS, "src/base.py", "Iface")
        impl_id = _add_symbol(g, NodeLabel.CLASS, "src/impl.py", "Impl")
        g.add_relationship(
            GraphRelationship(
                id="impl:1",
                type=RelType.IMPLEMENTS,
                source=impl_id,
                target=iface_id,
            )
        )
        assert _is_subclassed(g, iface_id, NodeLabel.CLASS) is True

    def test_class_without_subclasses(self) -> None:
        g = KnowledgeGraph()
        cls_id = _add_symbol(g, NodeLabel.CLASS, "src/leaf.py", "Leaf")
        assert _is_subclassed(g, cls_id, NodeLabel.CLASS) is False

    def test_function_not_subclassed(self) -> None:
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "src/utils.py", "foo")
        assert _is_subclassed(g, func_id, NodeLabel.FUNCTION) is False


# ---------------------------------------------------------------------------
# _is_dunder
# ---------------------------------------------------------------------------


class TestIsDunder:
    """_is_dunder matches dunder names."""

    def test_str(self) -> None:
        assert _is_dunder("__str__") is True

    def test_init(self) -> None:
        assert _is_dunder("__init__") is True

    def test_short(self) -> None:
        assert _is_dunder("____") is False

    def test_single_underscores(self) -> None:
        assert _is_dunder("_private_") is False


# ---------------------------------------------------------------------------
# Integration: process_dead_code with various predicates
# ---------------------------------------------------------------------------


class TestProcessDeadCodePredicates:
    """Integration tests for predicates through process_dead_code."""

    def test_html_file_symbols_exempt(self) -> None:
        """Symbols in HTML files are never flagged dead."""
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "public/index.html", "initPage")
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(func_id).is_dead is False

    def test_example_file_symbols_exempt(self) -> None:
        """Symbols in example files are never flagged dead."""
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "config/db.example.php", "getConfig")
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(func_id).is_dead is False

    def test_test_file_symbols_exempt(self) -> None:
        """Symbols in test files (helpers, fixtures) are never flagged dead."""
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "tests/conftest.py", "make_graph")
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(func_id).is_dead is False

    def test_framework_method_exempt(self) -> None:
        """Migration up/down methods are exempt from dead code."""
        g = KnowledgeGraph()
        up_id = _add_symbol(
            g, NodeLabel.METHOD, "database/migrations/2024_users.php", "up",
            class_name="CreateUsers",
        )
        down_id = _add_symbol(
            g, NodeLabel.METHOD, "database/migrations/2024_users.php", "down",
            class_name="CreateUsers",
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(up_id).is_dead is False
        assert g.get_node(down_id).is_dead is False

    def test_property_decorator_exempt(self) -> None:
        """@property methods are exempt (accessed as attributes, not called)."""
        g = KnowledgeGraph()
        prop_id = _add_symbol(
            g, NodeLabel.METHOD, "src/models.py", "full_name",
            class_name="User",
            properties={"decorators": ["property"]},
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(prop_id).is_dead is False

    def test_enum_class_exempt(self) -> None:
        """Enum classes are exempt (members accessed via dot, not called)."""
        g = KnowledgeGraph()
        cls_id = _add_symbol(
            g, NodeLabel.CLASS, "src/enums.py", "Color",
            properties={"bases": ["Enum"]},
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(cls_id).is_dead is False

    def test_python_public_api_exempt(self) -> None:
        """Public symbols in __init__.py are exempt."""
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "mylib/__init__.py", "create_app")
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(func_id).is_dead is False

    def test_private_init_not_exempt(self) -> None:
        """Private symbols in __init__.py are NOT exempt."""
        g = KnowledgeGraph()
        func_id = _add_symbol(g, NodeLabel.FUNCTION, "mylib/__init__.py", "_setup")
        count = process_dead_code(g)
        assert count == 1
        assert g.get_node(func_id).is_dead is True

    def test_lifecycle_methods_exempt(self) -> None:
        """Test lifecycle methods (setUp, tearDown) are exempt."""
        g = KnowledgeGraph()
        setup_id = _add_symbol(
            g, NodeLabel.METHOD, "tests/test_auth.php", "setUp",
            class_name="AuthTest",
        )
        teardown_id = _add_symbol(
            g, NodeLabel.METHOD, "tests/test_auth.php", "tearDown",
            class_name="AuthTest",
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(setup_id).is_dead is False
        assert g.get_node(teardown_id).is_dead is False

    def test_php_magic_methods_exempt(self) -> None:
        """PHP magic methods (__call, __get, etc.) are exempt."""
        g = KnowledgeGraph()
        call_id = _add_symbol(
            g, NodeLabel.METHOD, "src/models/User.php", "__call",
            class_name="User",
        )
        get_id = _add_symbol(
            g, NodeLabel.METHOD, "src/models/User.php", "__get",
            class_name="User",
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(call_id).is_dead is False
        assert g.get_node(get_id).is_dead is False

    def test_implicit_interface_method_exempt(self) -> None:
        """Implicit interface methods (jsonSerialize, count, etc.) are exempt."""
        g = KnowledgeGraph()
        json_id = _add_symbol(
            g, NodeLabel.METHOD, "src/models/User.php", "jsonSerialize",
            class_name="User",
        )
        count_id = _add_symbol(
            g, NodeLabel.METHOD, "src/models/User.php", "count",
            class_name="User",
        )
        to_array_id = _add_symbol(
            g, NodeLabel.METHOD, "src/models/User.php", "toArray",
            class_name="User",
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(json_id).is_dead is False
        assert g.get_node(count_id).is_dead is False
        assert g.get_node(to_array_id).is_dead is False

    def test_subclassed_class_exempt(self) -> None:
        """Classes with incoming EXTENDS edges are not dead."""
        g = KnowledgeGraph()
        base_id = _add_symbol(g, NodeLabel.CLASS, "src/base.py", "BaseModel")
        child_id = _add_symbol(g, NodeLabel.CLASS, "src/user.py", "User")
        g.add_relationship(
            GraphRelationship(
                id="ext:1",
                type=RelType.EXTENDS,
                source=child_id,
                target=base_id,
            )
        )
        # User extends BaseModel, but User itself is dead (no calls, no subclasses)
        count = process_dead_code(g)
        assert g.get_node(base_id).is_dead is False
        # User is dead (no incoming calls, not subclassed)
        assert g.get_node(child_id).is_dead is True
        assert count == 1

    def test_override_unflag(self) -> None:
        """Override methods are unflagged when base class method is called."""
        g = KnowledgeGraph()
        # Parent class with called method
        parent_cls_id = _add_symbol(g, NodeLabel.CLASS, "src/base.py", "Base")
        parent_method_id = _add_symbol(
            g, NodeLabel.METHOD, "src/base.py", "process",
            class_name="Base",
        )
        # Child class overriding method
        child_cls_id = _add_symbol(g, NodeLabel.CLASS, "src/child.py", "Child")
        child_method_id = _add_symbol(
            g, NodeLabel.METHOD, "src/child.py", "process",
            class_name="Child",
        )
        # EXTENDS relationship
        g.add_relationship(
            GraphRelationship(
                id="ext:1",
                type=RelType.EXTENDS,
                source=child_cls_id,
                target=parent_cls_id,
            )
        )
        # Caller calls parent method
        caller_id = _add_symbol(
            g, NodeLabel.FUNCTION, "src/main.py", "main",
            is_entry_point=True,
        )
        g.add_relationship(
            GraphRelationship(
                id="calls:1",
                type=RelType.CALLS,
                source=caller_id,
                target=parent_method_id,
            )
        )

        count = process_dead_code(g)
        # Child.process should be unflagged by the override pass
        assert g.get_node(child_method_id).is_dead is False

    def test_protocol_stub_unflag(self) -> None:
        """Protocol class stubs are unflagged."""
        g = KnowledgeGraph()
        proto_id = _add_symbol(
            g, NodeLabel.CLASS, "src/base.py", "Handler",
            properties={"is_protocol": True},
        )
        stub_id = _add_symbol(
            g, NodeLabel.METHOD, "src/base.py", "process",
            class_name="Handler",
        )

        count = process_dead_code(g)
        # Protocol stubs should not be flagged dead
        assert g.get_node(stub_id).is_dead is False

    def test_phpunit_test_method_exempt(self) -> None:
        """PHPUnit-style testMethodName is exempt."""
        g = KnowledgeGraph()
        test_id = _add_symbol(
            g, NodeLabel.METHOD, "src/auth.php", "testLoginReturnsToken",
            class_name="AuthTest",
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(test_id).is_dead is False

    def test_pytest_class_exempt(self) -> None:
        """pytest-style TestClassName is exempt."""
        g = KnowledgeGraph()
        cls_id = _add_symbol(g, NodeLabel.CLASS, "src/test_auth.py", "TestLoginFlow")
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(cls_id).is_dead is False

    def test_abstractmethod_exempt(self) -> None:
        """@abstractmethod stubs are not dead."""
        g = KnowledgeGraph()
        abs_id = _add_symbol(
            g, NodeLabel.METHOD, "src/base.py", "process",
            class_name="BaseHandler",
            properties={"decorators": ["abstractmethod"]},
        )
        count = process_dead_code(g)
        assert count == 0
        assert g.get_node(abs_id).is_dead is False

    def test_framework_decorator_names_exempt(self) -> None:
        """Named framework decorators (task, fixture, route, etc.) exempt."""
        g = KnowledgeGraph()
        task_id = _add_symbol(
            g, NodeLabel.FUNCTION, "src/tasks.py", "send_email",
            properties={"decorators": ["task"]},
        )
        fixture_id = _add_symbol(
            g, NodeLabel.FUNCTION, "src/conftest.py", "db_session",
            properties={"decorators": ["fixture"]},
        )
        count = process_dead_code(g)
        # send_email has @task (framework decorator name)
        assert g.get_node(task_id).is_dead is False
        # db_session has @fixture AND is in conftest.py (test file) — double exempt
        assert g.get_node(fixture_id).is_dead is False
