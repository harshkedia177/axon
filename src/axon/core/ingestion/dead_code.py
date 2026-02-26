"""Phase 10: Dead code detection for Axon.

Scans the knowledge graph to find unreachable symbols (functions, methods,
classes) that have zero incoming CALLS relationships and are not entry points,
exported, constructors, test functions, or dunder methods.  Flags them by
setting ``is_dead = True`` on the corresponding graph node.
"""

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

_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"__init__", "__new__", "__construct"})

# Test framework lifecycle methods invoked implicitly by the runner.
_LIFECYCLE_METHODS: frozenset[str] = frozenset({
    # PHPUnit / JUnit
    "setUp", "tearDown", "setUpBeforeClass", "tearDownAfterClass",
    # Python unittest
    "setUpClass", "tearDownClass", "setUpModule", "tearDownModule",
})

# PHP magic methods that are invoked implicitly by the runtime.
_PHP_MAGIC_METHODS: frozenset[str] = frozenset({
    "__construct", "__destruct", "__call", "__callStatic", "__get",
    "__set", "__isset", "__unset", "__sleep", "__wakeup", "__serialize",
    "__unserialize", "__toString", "__invoke", "__set_state", "__clone",
    "__debugInfo",
})

# Interface/protocol methods invoked implicitly by the runtime or framework.
# These methods are never called directly in user code but are triggered
# by language constructs (e.g. json_encode calls jsonSerialize, foreach
# calls getIterator/current/key/next/rewind/valid).
_IMPLICIT_INTERFACE_METHODS: frozenset[str] = frozenset({
    # PHP JsonSerializable
    "jsonSerialize",
    # PHP Countable
    "count",
    # PHP ArrayAccess
    "offsetGet", "offsetSet", "offsetExists", "offsetUnset",
    # PHP IteratorAggregate / Iterator
    "getIterator", "current", "key", "next", "rewind", "valid",
    # PHP Serializable
    "serialize", "unserialize",
    # PHP Stringable (invoked by string casts)
    "__toString",
    # Laravel / common conventions
    "toArray", "toJson",
    # JavaScript/TypeScript (called by JSON.stringify)
    "toJSON",
})

# Framework directory patterns where methods are invoked implicitly by the
# framework (e.g. Laravel migrations, seeders, console commands, event
# listeners, service providers, middleware, jobs).  Each tuple is
# (directory_substring, frozenset_of_method_names).
_FRAMEWORK_DIR_METHODS: tuple[tuple[str, frozenset[str]], ...] = (
    # Laravel / generic migrations
    ("/migrations/", frozenset({"up", "down"})),
    ("/database/migrations/", frozenset({"up", "down"})),
    # Seeders
    ("/seeders/", frozenset({"run"})),
    ("/seeds/", frozenset({"run"})),
    # Console commands
    ("/Commands/", frozenset({"handle"})),
    ("/commands/", frozenset({"handle"})),
    # Event listeners
    ("/Listeners/", frozenset({"handle"})),
    ("/listeners/", frozenset({"handle"})),
    # Service providers
    ("/Providers/", frozenset({"register", "boot"})),
    # Middleware
    ("/Middleware/", frozenset({"handle"})),
    ("/middleware/", frozenset({"handle"})),
    # Jobs / queue workers
    ("/Jobs/", frozenset({"handle"})),
    ("/jobs/", frozenset({"handle"})),
)


def _is_framework_method(name: str, file_path: str) -> bool:
    """Return ``True`` if *name* is a framework-invoked method in a framework directory.

    Methods like ``up()``/``down()`` in migration files, ``handle()`` in
    command/listener files, ``register()``/``boot()`` in service providers,
    etc. are invoked by the framework, never directly by user code.
    """
    for dir_pattern, method_names in _FRAMEWORK_DIR_METHODS:
        if dir_pattern in file_path and name in method_names:
            return True
    return False


def _is_test_method(name: str) -> bool:
    """Return ``True`` if *name* follows PHPUnit / JUnit test convention.

    Matches camelCase names starting with ``test`` where the next character
    is uppercase, e.g. ``testSanitizeEscapesHtml``, ``testEncryptRoundTrip``.
    """
    return len(name) > 4 and name.startswith("test") and name[4].isupper()

def _is_test_class(name: str) -> bool:
    """Return ``True`` if *name* follows pytest class convention (``Test*``).

    Matches names starting with ``Test`` where the next character is uppercase,
    e.g. ``TestHandleQuery``, ``TestBulkLoad``.
    """
    return len(name) > 4 and name.startswith("Test") and name[4].isupper()

def _is_test_file(file_path: str) -> bool:
    """Return ``True`` if the file is in a test directory or is a test file.

    Matches Python conventions (``/tests/``, ``test_*.py``, ``conftest.py``),
    JavaScript/TypeScript conventions (``__tests__/``, ``*.test.*``,
    ``*.spec.*``), and PHP conventions (``tests/`` or ``Tests/``,
    ``*Test.php``).
    """
    return (
        "/tests/" in file_path
        or "/Tests/" in file_path
        or file_path.startswith("tests/")
        or file_path.startswith("Tests/")
        or "/test/" in file_path
        or "/__tests__/" in file_path
        or "/test_" in file_path
        or ".test." in file_path
        or ".spec." in file_path
        or file_path.endswith("conftest.py")
        or file_path.endswith("Test.php")
    )

def _is_example_file(file_path: str) -> bool:
    """Return ``True`` if the file is an example/template/sample file.

    Files like ``config.example.php`` or ``settings.sample.py`` are
    documentation templates, not production code.  Symbols in them should
    never be flagged as dead code.
    """
    return (
        ".example." in file_path
        or ".sample." in file_path
        or ".template." in file_path
        or "/examples/" in file_path
        or "/example/" in file_path
    )

def _is_html_file(file_path: str) -> bool:
    """Return ``True`` if the file is an HTML file.

    Functions in HTML inline ``<script>`` blocks are almost always page-local
    UI handlers bound via template-literal ``onclick`` strings, ``addEventListener``
    callbacks, or array references that cannot be statically extracted.
    """
    return file_path.endswith((".html", ".htm"))

def _is_dunder(name: str) -> bool:
    """Return ``True`` if *name* is a dunder (double-underscore) method.

    Dunders start and end with ``__`` and have at least one character in
    between (e.g. ``__str__``, ``__repr__``).
    """
    return name.startswith("__") and name.endswith("__") and len(name) > 4

def _is_type_referenced(graph: KnowledgeGraph, node_id: str, label: NodeLabel) -> bool:
    """Return ``True`` if *node_id* is a class with incoming USES_TYPE edges.

    Classes referenced via type annotations (enums, dataclasses, Protocol
    classes) are not dead — they are actively used as types.  This check
    is restricted to CLASS nodes; a function used only in a type annotation
    is legitimately unused.
    """
    if label != NodeLabel.CLASS:
        return False
    return graph.has_incoming(node_id, RelType.USES_TYPE)


def _is_subclassed(graph: KnowledgeGraph, node_id: str, label: NodeLabel) -> bool:
    """Return ``True`` if *node_id* is a class or interface with subclasses.

    A class that has incoming EXTENDS or IMPLEMENTS edges is being
    subclassed/implemented — it is not dead even if never instantiated
    directly (e.g. abstract base classes, interfaces).
    """
    if label != NodeLabel.CLASS:
        return False
    return (
        graph.has_incoming(node_id, RelType.EXTENDS)
        or graph.has_incoming(node_id, RelType.IMPLEMENTS)
    )

_NON_FRAMEWORK_DECORATORS: frozenset[str] = frozenset({
    "functools.wraps",
    "functools.lru_cache",
    "functools.cached_property",
    "functools.cache",
})

_FRAMEWORK_DECORATOR_NAMES: frozenset[str] = frozenset({
    "task", "shared_task", "periodic_task", "job",
    "receiver", "on_event", "handler",
    "validator", "field_validator", "root_validator", "model_validator",
    "contextmanager", "asynccontextmanager",
    "fixture",
    "route", "endpoint", "command",
    "hybrid_property",
})

def _has_framework_decorator(node: GraphNode) -> bool:
    """Return ``True`` if *node* has a framework decorator (dotted or undotted)."""
    decorators: list[str] = node.properties.get("decorators", [])
    return any(
        dec in _FRAMEWORK_DECORATOR_NAMES or ("." in dec and dec not in _NON_FRAMEWORK_DECORATORS)
        for dec in decorators
    )

def _has_property_decorator(node: GraphNode) -> bool:
    """Return ``True`` if *node* is a ``@property`` (accessed as attribute, not called)."""
    decorators: list[str] = node.properties.get("decorators", [])
    return "property" in decorators

_TYPING_STUB_DECORATORS: frozenset[str] = frozenset({
    "overload", "typing.overload",
    "abstractmethod", "abc.abstractmethod",
})

def _has_typing_stub_decorator(node: GraphNode) -> bool:
    """Return ``True`` if *node* is an ``@overload`` or ``@abstractmethod`` stub."""
    decorators: list[str] = node.properties.get("decorators", [])
    return any(d in _TYPING_STUB_DECORATORS for d in decorators)

_ENUM_BASES: frozenset[str] = frozenset({
    "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
})

def _is_enum_class(node: GraphNode, label: NodeLabel) -> bool:
    """Return ``True`` if *node* is an enum class (members accessed via dot, not called)."""
    if label != NodeLabel.CLASS:
        return False
    bases: list[str] = node.properties.get("bases", [])
    return bool(_ENUM_BASES & set(bases))

def _is_python_public_api(name: str, file_path: str) -> bool:
    """Return ``True`` if *name* is a public symbol in an ``__init__.py`` file."""
    return file_path.endswith("__init__.py") and not name.startswith("_")

def _is_exempt(
    name: str, is_entry_point: bool, is_exported: bool, file_path: str = ""
) -> bool:
    """Return ``True`` if the symbol is exempt from dead-code flagging.

    A symbol is exempt when ANY of the following hold:

    - It is marked as an entry point.
    - It is marked as exported (may be used externally).
    - It is a constructor (``__init__`` / ``__new__``).
    - It is a test function (name starts with ``test_``).
    - It is a test class (name starts with ``Test``).
    - It lives in a test file (fixtures, helpers are not dead code).
    - It is a dunder method (``__str__``, ``__repr__``, etc.).
    - It is a public symbol in a Python ``__init__.py`` file.
    - It is a test lifecycle method (``setUp``, ``tearDown``, etc.).
    - It lives in an HTML file (inline script functions are page-local).
    - It lives in an example/template file (documentation, not production).
    """
    return (
        is_entry_point
        or is_exported
        or name in _CONSTRUCTOR_NAMES
        or name in _PHP_MAGIC_METHODS
        or name in _LIFECYCLE_METHODS
        or name.startswith("test_")
        or _is_test_method(name)
        or _is_test_class(name)
        or _is_test_file(file_path)
        or _is_dunder(name)
        or _is_python_public_api(name, file_path)
        or _is_html_file(file_path)
        or _is_example_file(file_path)
        or _is_framework_method(name, file_path)
    )

def _clear_override_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods that override a non-dead base class method.

    When ``A extends B`` and ``B.method`` is called, ``A.method`` (the
    override) has zero incoming CALLS and gets flagged dead.  This pass
    detects that situation and clears ``is_dead`` on the override.

    Returns the number of overrides un-flagged.
    """
    # Build a mapping: class_name -> set of method names that are NOT dead.
    alive_methods_by_class: dict[str, set[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead and method.class_name:
            alive_methods_by_class.setdefault(method.class_name, set()).add(method.name)

    # Build child -> parent class mapping from EXTENDS and IMPLEMENTS relationships.
    child_to_parents: dict[str, list[str]] = {}
    for rel_type in (RelType.EXTENDS, RelType.IMPLEMENTS):
        for rel in graph.get_relationships_by_type(rel_type):
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
    """Un-flag methods on classes that structurally conform to a Protocol.

    When a Protocol defines methods ``{m1, m2, m3}`` and a concrete class
    implements all of those methods without an explicit EXTENDS edge
    (structural subtyping), the concrete methods may be flagged dead
    because CALLS edges resolve to the Protocol's stubs, not the
    concrete implementations.

    This pass:

    1. Finds Protocol classes (annotated with ``is_protocol`` in properties).
    2. Collects their non-dunder method names as the required interface.
    3. Finds non-Protocol classes whose methods are a superset.
    4. Un-flags dead methods whose name is in the protocol interface.

    Returns the number of methods un-flagged.
    """
    protocol_methods: dict[str, set[str]] = {}
    for cls_node in graph.get_nodes_by_label(NodeLabel.CLASS):
        if not cls_node.properties.get("is_protocol"):
            continue
        methods = set()
        for method in graph.get_nodes_by_label(NodeLabel.METHOD):
            if method.class_name == cls_node.name and not _is_dunder(method.name):
                methods.add(method.name)
        if methods:
            protocol_methods[cls_node.name] = methods

    if not protocol_methods:
        return 0

    class_methods: dict[str, set[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if method.class_name:
            class_methods.setdefault(method.class_name, set()).add(method.name)

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
    """Un-flag methods on Protocol classes.

    Protocol stubs define the interface contract — they are never called
    directly (calls resolve to concrete implementations).  Flagging them
    as dead is always a false positive.

    Returns the number of methods un-flagged.
    """
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
            if _is_subclassed(graph, node.id, label):
                continue
            if _has_framework_decorator(node):
                continue
            if _has_property_decorator(node):
                continue
            if _has_typing_stub_decorator(node):
                continue
            if _is_enum_class(node, label):
                continue
            if label == NodeLabel.METHOD and node.name in _IMPLICIT_INTERFACE_METHODS:
                continue

            node.is_dead = True
            dead_count += 1
            logger.debug("Dead symbol: %s (%s)", node.name, node.id)

    # Second pass: un-flag overrides of called base-class methods.
    cleared = _clear_override_false_positives(graph)
    dead_count -= cleared

    # Third pass: un-flag methods on classes that structurally conform to a Protocol.
    protocol_cleared = _clear_protocol_conformance_false_positives(graph)
    dead_count -= protocol_cleared

    # Fourth pass: un-flag Protocol class stubs (interface contracts, never called directly).
    stub_cleared = _clear_protocol_stub_false_positives(graph)
    dead_count -= stub_cleared

    return dead_count
