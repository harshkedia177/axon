"""Phase 5: Call tracing for Axon.

Takes FileParseData from the parser phase and resolves call expressions to
target symbol nodes, creating CALLS relationships with confidence scores.

Resolution priority:
1. Same-file exact match (confidence 1.0)
2. Import-resolved match (confidence 1.0)
3. Global fuzzy match (confidence 0.5)
4. Receiver method resolution (confidence 0.8)
"""

from __future__ import annotations

import logging

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import build_file_symbol_index, build_name_index, find_containing_symbol
from axon.core.parsers.base import CallInfo

logger = logging.getLogger(__name__)

_CALLABLE_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)

_KIND_TO_LABEL: dict[str, NodeLabel] = {
    "function": NodeLabel.FUNCTION,
    "method": NodeLabel.METHOD,
    "class": NodeLabel.CLASS,
}

# Names that should never produce CALLS edges.  These are language builtins,
# stdlib utilities, framework hooks, and common JS/TS globals whose definitions
# do not exist in the user's codebase.  Filtering them before resolution
# prevents low-confidence global-fuzzy matches against short, common names.
_CALL_BLOCKLIST: frozenset[str] = frozenset({
    # Python builtins
    "print", "len", "range", "map", "filter", "sorted", "list", "dict",
    "set", "str", "int", "float", "bool", "type", "super", "isinstance",
    "issubclass", "hasattr", "getattr", "setattr", "open", "iter", "next",
    "zip", "enumerate", "any", "all", "min", "max", "sum", "abs", "round",
    "repr", "id", "hash", "dir", "vars", "input", "format", "tuple",
    "frozenset", "bytes", "bytearray", "memoryview", "object", "property",
    "classmethod", "staticmethod", "delattr", "callable", "compile", "eval",
    "exec", "globals", "locals", "breakpoint", "exit", "quit",
    # Python stdlib — common method names that collide with user-defined symbols
    "append", "extend", "update", "pop", "get", "items", "keys", "values",
    "split", "join", "strip", "replace", "startswith", "endswith", "lower",
    "upper", "encode", "decode", "read", "write", "close",
    # JS/TS built-in globals
    "console", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "JSON", "Array", "Object", "Promise", "Math", "Date", "Error", "Symbol",
    "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURIComponent",
    "decodeURIComponent", "fetch", "require", "exports", "module",
    "document", "window", "process", "Buffer", "URL",
    # JS/TS dotted method names extracted as bare call names
    "log", "error", "warn", "info", "debug",
    "parse", "stringify",
    "assign", "freeze",
    "isArray", "from", "of",
    "resolve", "reject", "race",
    "floor", "ceil", "random",
    # React hooks
    "useState", "useEffect", "useRef", "useCallback", "useMemo",
    "useContext", "useReducer", "useLayoutEffect", "useImperativeHandle",
    "useDebugValue", "useId", "useTransition", "useDeferredValue",
    # PHP builtins
    "echo", "print_r", "var_dump", "isset", "unset", "empty", "die", "exit",
    "array_map", "array_filter", "array_merge", "array_push", "array_pop",
    "array_shift", "array_unshift", "array_keys", "array_values",
    "array_unique", "array_reverse", "array_slice", "array_splice",
    "array_search", "array_key_exists", "in_array", "count", "sizeof",
    "strlen", "strpos", "substr", "strtolower", "strtoupper", "trim",
    "ltrim", "rtrim", "explode", "implode", "sprintf", "printf",
    "str_replace", "preg_match", "preg_replace", "preg_match_all",
    "is_null", "is_array", "is_string", "is_int", "is_numeric", "is_bool",
    "intval", "floatval", "strval", "boolval",
    "json_encode", "json_decode", "date", "time", "strtotime",
    "file_get_contents", "file_put_contents", "fopen", "fclose", "fread",
    "fwrite", "file_exists", "is_file", "is_dir", "mkdir", "unlink",
    "header", "session_start", "session_destroy",
    "defined", "define", "constant", "class_exists", "method_exists",
    "function_exists", "property_exists", "get_class", "get_parent_class",
    "compact", "extract", "array_combine", "range", "usort", "uksort",
    "array_walk", "array_column", "array_fill", "array_chunk",
    "htmlspecialchars", "htmlentities", "urlencode", "urldecode",
    "base64_encode", "base64_decode", "md5", "sha1", "hash",
    "number_format", "round", "ceil", "floor", "abs", "max", "min",
    "rand", "mt_rand", "array_rand",
})

def resolve_call(
    call: CallInfo,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
) -> tuple[str | None, float]:
    """Resolve a call expression to a target node ID and confidence score.

    Resolution strategy (tried in order):

    1. **Same-file exact match** (confidence 1.0) -- the called symbol is
       defined in the same file as the caller.
    2. **Import-resolved match** (confidence 1.0) -- the called name was
       imported into this file; find the symbol in the imported file.
    3. **Global fuzzy match** (confidence 0.5) -- any symbol with this name
       anywhere in the codebase.  If multiple matches exist, the one with
       the shortest file path is chosen (heuristic for proximity).

    For method calls (``call.receiver`` is non-empty):
    - If the receiver is ``"self"`` or ``"this"``, look for a method with
      that name in the same class (same file, matching class_name).
    - Otherwise, try to resolve the method name globally.

    Args:
        call: The parsed call information.
        file_path: Path to the file containing the call.
        call_index: Mapping from symbol names to node IDs built by
            :func:`build_call_index`.
        graph: The knowledge graph.

    Returns:
        A tuple of ``(node_id, confidence)`` or ``(None, 0.0)`` if the
        call cannot be resolved.
    """
    name = call.name
    receiver = call.receiver

    if receiver in ("self", "this"):
        result = _resolve_self_method(name, file_path, call_index, graph)
        if result is not None:
            return result, 1.0

    # Without type info the receiver doesn't help — fall through to name-based resolution.
    candidate_ids = call_index.get(name, [])
    if not candidate_ids:
        return None, 0.0

    # 1. Same-file exact match.
    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path == file_path:
            return nid, 1.0

    # 2. Import-resolved match.
    imported_target = _resolve_via_imports(name, file_path, candidate_ids, graph)
    if imported_target is not None:
        return imported_target, 1.0

    # 3. Global fuzzy match -- prefer shortest file path.
    return _pick_closest(candidate_ids, graph), 0.5

def _resolve_self_method(
    method_name: str,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
) -> str | None:
    """Find a method with *method_name* in the same file (same class).

    When the receiver is ``self`` or ``this`` the target must be a Method
    node defined in the same file.
    """
    for nid in call_index.get(method_name, []):
        node = graph.get_node(nid)
        if (
            node is not None
            and node.label == NodeLabel.METHOD
            and node.file_path == file_path
        ):
            return nid
    return None

def _resolve_via_imports(
    name: str,
    file_path: str,
    candidate_ids: list[str],
    graph: KnowledgeGraph,
) -> str | None:
    """Check if *name* was imported into *file_path* and resolve to the target.

    Looks at IMPORTS relationships originating from this file's File node.
    For each imported file, checks whether any candidate symbol is defined
    there.  Also checks the ``symbols`` property to see if the specific
    name was explicitly imported.
    """
    source_file_id = generate_id(NodeLabel.FILE, file_path)
    import_rels = graph.get_outgoing(source_file_id, RelType.IMPORTS)

    if not import_rels:
        return None

    # Collect file paths of imported files, optionally filtering by
    # the imported symbol names.
    imported_file_ids: set[str] = set()
    for rel in import_rels:
        symbols_str = rel.properties.get("symbols", "")
        imported_names = {s.strip() for s in symbols_str.split(",") if s.strip()}

        # If the specific name was imported, or if it's a wildcard/full
        # module import (no specific names), include this target file.
        if not imported_names or name in imported_names:
            target_node = graph.get_node(rel.target)
            if target_node is not None:
                imported_file_ids.add(target_node.file_path)

    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path in imported_file_ids:
            return nid

    return None

def _pick_closest(candidate_ids: list[str], graph: KnowledgeGraph) -> str | None:
    """Pick the candidate with the shortest file path (proximity heuristic).

    Returns ``None`` if no candidates can be resolved to actual nodes.
    """
    best_id: str | None = None
    best_path_len = float("inf")

    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and len(node.file_path) < best_path_len:
            best_path_len = len(node.file_path)
            best_id = nid

    return best_id

def _add_calls_edge(
    source_id: str,
    target_id: str,
    confidence: float,
    graph: KnowledgeGraph,
    seen: set[str],
) -> None:
    """Create a deduplicated CALLS relationship."""
    rel_id = f"calls:{source_id}->{target_id}"
    if rel_id not in seen:
        seen.add(rel_id)
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.CALLS,
                source=source_id,
                target=target_id,
                properties={"confidence": confidence},
            )
        )

def _resolve_receiver_method(
    receiver: str,
    method_name: str,
    source_id: str,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    seen: set[str],
) -> None:
    """Resolve ``Receiver.method()`` to the METHOD node and create a CALLS edge.

    Looks for a METHOD node whose ``name`` matches *method_name* and whose
    ``class_name`` matches *receiver*.  Searches same-file first, then
    globally.
    """
    same_file_match: str | None = None
    global_match: str | None = None

    for nid in call_index.get(method_name, []):
        node = graph.get_node(nid)
        if (
            node is not None
            and node.label == NodeLabel.METHOD
            and node.class_name == receiver
        ):
            if node.file_path == file_path:
                same_file_match = nid
                break
            elif global_match is None:
                global_match = nid

    target = same_file_match or global_match
    if target is not None:
        _add_calls_edge(source_id, target, 0.8, graph, seen)


def process_calls(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> None:
    """Resolve call expressions and create CALLS relationships in the graph.

    For each call expression in the parse data:

    1. Determine which symbol in the file *contains* the call (by line
       number range).
    2. Resolve the call to a target symbol node.
    3. Create a CALLS relationship from the containing symbol to the
       target, with a ``confidence`` property.

    Skips calls where:
    - The containing symbol cannot be determined.
    - The target cannot be resolved.
    - A relationship with the same ID already exists (deduplication).

    Args:
        parse_data: File parse results from the parser phase.
        graph: The knowledge graph to populate with CALLS relationships.
    """
    call_index = build_name_index(graph, _CALLABLE_LABELS)
    file_sym_index = build_file_symbol_index(graph, _CALLABLE_LABELS)
    seen: set[str] = set()

    for fpd in parse_data:
        for call in fpd.parse_result.calls:
            is_blocklisted = (
                call.name in _CALL_BLOCKLIST
                and call.receiver not in ("self", "this")
            )

            source_id = find_containing_symbol(
                call.line, fpd.file_path, file_sym_index
            )
            if source_id is None:
                # Top-level calls (e.g. PHP switch-dispatch, HTML inline
                # scripts, JS module-level code) have no containing symbol.
                # Fall back to the FILE node so the target still gets an
                # incoming CALLS edge and avoids false dead-code flags.
                source_id = generate_id(NodeLabel.FILE, fpd.file_path)
                if graph.get_node(source_id) is None:
                    continue

            if not is_blocklisted:
                target_id, confidence = resolve_call(
                    call, fpd.file_path, call_index, graph
                )
                if target_id is not None:
                    _add_calls_edge(source_id, target_id, confidence, graph, seen)

            # Callback arguments: bare identifiers passed as arguments
            # (e.g. map(transform, items), Depends(get_db),
            # setTimeout(handler, 1000)).  Always processed even when the
            # callee is blocklisted — the callback itself may be user code.
            for arg_name in call.arguments:
                if arg_name in _CALL_BLOCKLIST:
                    continue
                arg_call = CallInfo(name=arg_name, line=call.line)
                arg_id, arg_conf = resolve_call(
                    arg_call, fpd.file_path, call_index, graph
                )
                if arg_id is not None:
                    _add_calls_edge(source_id, arg_id, arg_conf * 0.8, graph, seen)

            # Receiver: link to the class and resolve the method on it.
            if not is_blocklisted:
                receiver = call.receiver
                if receiver and receiver not in ("self", "this"):
                    receiver_call = CallInfo(name=receiver, line=call.line)
                    recv_id, recv_conf = resolve_call(
                        receiver_call, fpd.file_path, call_index, graph
                    )
                    if recv_id is not None:
                        _add_calls_edge(source_id, recv_id, recv_conf, graph, seen)

                    _resolve_receiver_method(
                        receiver, call.name, source_id, fpd.file_path,
                        call_index, graph, seen,
                    )

        # Decorators are implicit calls — @cost_decorator on a function is
        # equivalent to calling cost_decorator(func).  Create CALLS edges
        # from the decorated symbol to the decorator definition.
        for symbol in fpd.parse_result.symbols:
            if not symbol.decorators:
                continue

            symbol_name = (
                f"{symbol.class_name}.{symbol.name}"
                if symbol.kind == "method" and symbol.class_name
                else symbol.name
            )
            label = _KIND_TO_LABEL.get(symbol.kind)
            if label is None:
                continue
            source_id = generate_id(label, fpd.file_path, symbol_name)

            for dec_name in symbol.decorators:
                # Strip the base name for dotted decorators (e.g. "app.route" → "route")
                # but also try the full dotted name.
                base_name = dec_name.rsplit(".", 1)[-1] if "." in dec_name else dec_name
                call_obj = CallInfo(name=base_name, line=symbol.start_line)
                target_id, confidence = resolve_call(
                    call_obj, fpd.file_path, call_index, graph
                )
                if target_id is None and "." in dec_name:
                    # Try full dotted name as well.
                    call_obj = CallInfo(name=dec_name, line=symbol.start_line)
                    target_id, confidence = resolve_call(
                        call_obj, fpd.file_path, call_index, graph
                    )
                if target_id is None:
                    continue

                rel_id = f"calls:{source_id}->{target_id}"
                if rel_id in seen:
                    continue
                seen.add(rel_id)

                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=RelType.CALLS,
                        source=source_id,
                        target=target_id,
                        properties={"confidence": confidence},
                    )
                )
