"""Phase 10: Dead code detection for Axon."""

from __future__ import annotations

import logging

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, NodeLabel, RelType

logger = logging.getLogger(__name__)

_SYMBOL_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)

_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"__init__", "__new__"})


def _is_test_class(name: str) -> bool:
    return len(name) > 4 and name.startswith("Test") and name[4].isupper()


def _is_test_file(file_path: str) -> bool:
    parts = file_path.replace("\\", "/").split("/")
    return (
        "tests" in parts
        or "test" in parts
        or any(p.startswith("test_") for p in parts)
        or file_path.endswith("conftest.py")
    )


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _is_type_referenced(graph: KnowledgeGraph, node_id: str, label: NodeLabel) -> bool:
    """Return True if node_id is a CLASS with incoming USES_TYPE or EXTENDS edges."""
    if label != NodeLabel.CLASS:
        return False
    return graph.has_incoming(node_id, RelType.USES_TYPE) or graph.has_incoming(
        node_id, RelType.EXTENDS
    )


_NON_FRAMEWORK_DECORATORS: frozenset[str] = frozenset(
    {
        "functools.wraps",
        "functools.lru_cache",
        "functools.cached_property",
        "functools.cache",
    }
)

_FRAMEWORK_DECORATOR_NAMES: frozenset[str] = frozenset(
    {
        "task",
        "shared_task",
        "periodic_task",
        "job",
        "receiver",
        "on_event",
        "handler",
        "validator",
        "field_validator",
        "root_validator",
        "model_validator",
        "computed_field",
        "model_serializer",
        "field_serializer",
        "contextmanager",
        "asynccontextmanager",
        "fixture",
        "route",
        "endpoint",
        "command",
        "hybrid_property",
    }
)


def _has_framework_decorator(node: GraphNode) -> bool:
    decorators: list[str] = node.properties.get("decorators", [])
    return any(
        dec in _FRAMEWORK_DECORATOR_NAMES or ("." in dec and dec not in _NON_FRAMEWORK_DECORATORS)
        for dec in decorators
    )


def _has_property_decorator(node: GraphNode) -> bool:
    decorators: list[str] = node.properties.get("decorators", [])
    return "property" in decorators


_TYPING_STUB_DECORATORS: frozenset[str] = frozenset(
    {
        "overload",
        "typing.overload",
        "abstractmethod",
        "abc.abstractmethod",
    }
)


def _has_typing_stub_decorator(node: GraphNode) -> bool:
    decorators: list[str] = node.properties.get("decorators", [])
    return any(d in _TYPING_STUB_DECORATORS for d in decorators)


_ENUM_BASES: frozenset[str] = frozenset(
    {
        "Enum",
        "IntEnum",
        "StrEnum",
        "Flag",
        "IntFlag",
    }
)


def _is_enum_class(node: GraphNode, label: NodeLabel) -> bool:
    if label != NodeLabel.CLASS:
        return False
    bases: list[str] = node.properties.get("bases", [])
    return bool(_ENUM_BASES & set(bases))


def _is_alembic_migration(file_path: str) -> bool:
    """Return True if *file_path* looks like an Alembic migration file."""
    parts = file_path.replace("\\", "/").split("/")
    return "versions" in parts or "migrations" in parts


def _is_python_public_api(name: str, file_path: str) -> bool:
    return file_path.endswith("__init__.py") and not name.startswith("_")


def _is_exempt(name: str, is_entry_point: bool, is_exported: bool, file_path: str = "") -> bool:
    return (
        is_entry_point
        or is_exported
        or name in _CONSTRUCTOR_NAMES
        or name.startswith("test_")
        or _is_test_class(name)
        or _is_test_file(file_path)
        or _is_dunder(name)
        or _is_python_public_api(name, file_path)
        or (name in ("upgrade", "downgrade") and _is_alembic_migration(file_path))
    )


def _clear_override_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods that override a non-dead base class method."""
    alive_methods_by_class: dict[str, set[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead and method.class_name:
            alive_methods_by_class.setdefault(method.class_name, set()).add(method.name)

    child_to_parents: dict[str, list[str]] = {}
    for rel in graph.get_relationships_by_type(RelType.EXTENDS):
        child_node = graph.get_node(rel.source)
        parent_node = graph.get_node(rel.target)
        if child_node and parent_node:
            child_to_parents.setdefault(child_node.name, []).append(parent_node.name)

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue

        parent_classes = child_to_parents.get(method.class_name, [])
        for parent_name in parent_classes:
            alive_in_parent = alive_methods_by_class.get(parent_name, set())
            if method.name in alive_in_parent:
                method.is_dead = False
                cleared += 1
                logger.debug("Un-flagged override: %s.%s", method.class_name, method.name)
                break

    return cleared


def _clear_protocol_conformance_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods on classes that structurally conform to a Protocol."""
    class_methods: dict[str, set[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if method.class_name:
            class_methods.setdefault(method.class_name, set()).add(method.name)

    protocol_methods: dict[str, set[str]] = {}
    for cls_node in graph.get_nodes_by_label(NodeLabel.CLASS):
        if not cls_node.properties.get("is_protocol"):
            continue
        methods = class_methods.get(cls_node.name, set())
        non_dunder = {m for m in methods if not _is_dunder(m)}
        if non_dunder:
            protocol_methods[cls_node.name] = non_dunder

    if not protocol_methods:
        return 0

    clearable: dict[str, set[str]] = {}
    for proto_name, required in protocol_methods.items():
        for cls_name, methods in class_methods.items():
            if cls_name == proto_name:
                continue
            if required <= methods:  # structural conformance
                clearable.setdefault(cls_name, set()).update(required)

    if not clearable:
        return 0

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue
        names_to_clear = clearable.get(method.class_name)
        if names_to_clear and method.name in names_to_clear:
            method.is_dead = False
            cleared += 1
            logger.debug(
                "Un-flagged protocol conformance: %s.%s",
                method.class_name,
                method.name,
            )

    return cleared


def _clear_protocol_stub_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods on Protocol classes (stubs are never called directly)."""
    protocol_class_names: set[str] = set()
    for cls_node in graph.get_nodes_by_label(NodeLabel.CLASS):
        if cls_node.properties.get("is_protocol"):
            protocol_class_names.add(cls_node.name)

    if not protocol_class_names:
        return 0

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue
        if method.class_name in protocol_class_names:
            method.is_dead = False
            cleared += 1
            logger.debug("Un-flagged protocol stub: %s.%s", method.class_name, method.name)

    return cleared


def process_dead_code(graph: KnowledgeGraph) -> int:
    """Detect dead (unreachable) symbols and flag them in the graph.

    A symbol is considered dead when **all** of the following are true:

    1. It has zero incoming ``CALLS`` relationships.
    2. It is not an entry point (``is_entry_point == False``).
    3. It is not exported (``is_exported == False``).
    4. It is not a class constructor (``__init__`` / ``__new__``).
    5. It is not a test function (name starts with ``test_``).
    6. It is not a test class (name starts with ``Test``).
    7. It is not in a test file (fixtures/helpers are exempt).
    8. It is not a dunder method (name starts and ends with ``__``).
    9. It is not a class referenced via type annotations (``USES_TYPE``).
    10. It does not have a framework-registration decorator.
    11. It is not a ``@property`` method.
    12. It is not an ``@overload`` or ``@abstractmethod`` stub.
    13. It is not an enum class (extends ``Enum``, ``IntEnum``, etc.).

    After the initial pass, three additional passes reduce false positives:

    - **Override pass**: un-flags method overrides whose base class method
      is called (resolves dynamic dispatch false positives).
    - **Protocol conformance pass**: un-flags methods on classes that
      structurally conform to a Protocol interface.
    - **Protocol stub pass**: un-flags methods on Protocol classes
      themselves (stubs are never called directly).

    For each dead symbol the function sets ``node.is_dead = True``.

    Args:
        graph: The knowledge graph to scan and mutate.

    Returns:
        The total number of symbols flagged as dead.
    """
    dead_count = 0

    for label in _SYMBOL_LABELS:
        for node in graph.get_nodes_by_label(label):
            if _is_exempt(node.name, node.is_entry_point, node.is_exported, node.file_path):
                continue
            if graph.has_incoming(node.id, RelType.CALLS):
                continue
            if _is_type_referenced(graph, node.id, label):
                continue
            if _has_framework_decorator(node):
                continue
            if _has_property_decorator(node):
                continue
            if _has_typing_stub_decorator(node):
                continue
            if _is_enum_class(node, label):
                continue

            node.is_dead = True
            dead_count += 1
            logger.debug("Dead symbol: %s (%s)", node.name, node.id)

    dead_count -= _clear_override_false_positives(graph)
    dead_count -= _clear_protocol_conformance_false_positives(graph)
    dead_count -= _clear_protocol_stub_false_positives(graph)

    return max(0, dead_count)
