"""Phase 4: Import resolution for Axon.

Takes the FileParseData produced by the parsing phase and resolves import
statements to actual File nodes in the knowledge graph, creating IMPORTS
relationships between the importing file and the target file.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.parsers.base import ImportInfo

logger = logging.getLogger(__name__)

_JS_TS_EXTENSIONS = (".ts", ".js", ".tsx", ".jsx", ".mjs", ".cjs")

def build_file_index(graph: KnowledgeGraph) -> dict[str, str]:
    """Build an index mapping file paths to their graph node IDs.

    Iterates over all :pyclass:`NodeLabel.FILE` nodes in the graph and
    returns a dict keyed by ``file_path`` with node ``id`` as value.

    Args:
        graph: The knowledge graph containing File nodes.

    Returns:
        A dict like ``{"src/auth/validate.py": "file:src/auth/validate.py:"}``.
    """
    file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)
    return {node.file_path: node.id for node in file_nodes}

def resolve_import_path(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve an import statement to the target file's node ID.

    Uses the importing file's path, the parsed :class:`ImportInfo`, and the
    index of all known project files to determine which file is being
    imported.  Returns ``None`` for external/unresolvable imports.

    Args:
        importing_file: Relative path of the file containing the import
            (e.g. ``"src/auth/validate.py"``).
        import_info: The parsed import information.
        file_index: Mapping of relative file paths to their graph node IDs.

    Returns:
        The node ID of the resolved target file, or ``None`` if the import
        cannot be resolved to a file in the project.
    """
    language = _detect_language(importing_file)

    if language == "python":
        return _resolve_python(importing_file, import_info, file_index)
    if language in ("typescript", "tsx", "javascript"):
        return _resolve_js_ts(importing_file, import_info, file_index)
    if language == "php":
        return _resolve_php(importing_file, import_info, file_index)
    if language == "html":
        return _resolve_html(importing_file, import_info, file_index)

    return None

def process_imports(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> None:
    """Resolve imports and create IMPORTS relationships in the graph.

    For each file's parsed imports, resolves the target file and creates
    an ``IMPORTS`` relationship from the importing file node to the target
    file node.  Duplicate edges (same source -> same target) are skipped.

    Args:
        parse_data: Parse results from the parsing phase.
        graph: The knowledge graph to populate with IMPORTS relationships.
    """
    file_index = build_file_index(graph)
    seen: set[tuple[str, str]] = set()

    for fpd in parse_data:
        source_file_id = generate_id(NodeLabel.FILE, fpd.file_path)

        for imp in fpd.parse_result.imports:
            target_id = resolve_import_path(fpd.file_path, imp, file_index)
            if target_id is None:
                continue

            pair = (source_file_id, target_id)
            if pair in seen:
                continue
            seen.add(pair)

            rel_id = f"imports:{source_file_id}->{target_id}"
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.IMPORTS,
                    source=source_file_id,
                    target=target_id,
                    properties={"symbols": ",".join(imp.names)},
                )
            )

def _detect_language(file_path: str) -> str:
    """Infer language from a file's extension."""
    suffix = PurePosixPath(file_path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".ts":
        return "typescript"
    if suffix == ".tsx":
        return "tsx"
    if suffix in (".js", ".jsx", ".mjs", ".cjs"):
        return "javascript"
    if suffix == ".php":
        return "php"
    if suffix in (".html", ".htm"):
        return "html"
    return ""

def _resolve_python(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a Python import to a file node ID.

    Handles:
    - Relative imports (``is_relative=True``): dot-prefixed module paths
      resolved relative to the importing file's directory.
    - Absolute imports: treated as dotted paths from the project root.

    Returns ``None`` for external (not in file_index) imports.
    """
    if import_info.is_relative:
        return _resolve_python_relative(importing_file, import_info, file_index)
    return _resolve_python_absolute(import_info, file_index)

def _resolve_python_relative(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a relative Python import (``from .foo import bar``).

    The number of leading dots determines how many directory levels to
    traverse upward from the importing file's parent directory.

    ``from .utils import helper``  -> one dot  -> same directory
    ``from ..models import User``  -> two dots -> parent directory
    """
    module = import_info.module

    dot_count = 0
    for ch in module:
        if ch == ".":
            dot_count += 1
        else:
            break

    remainder = module[dot_count:]

    base = PurePosixPath(importing_file).parent
    for _ in range(dot_count - 1):
        base = base.parent

    if remainder:
        segments = remainder.split(".")
        target_dir = base / PurePosixPath(*segments)
    else:
        target_dir = base

    return _try_python_paths(str(target_dir), file_index)

def _resolve_python_absolute(
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve an absolute Python import (``from mypackage.auth import validate``).

    Converts the dotted module path to a filesystem path and looks it up
    in the file index.  Returns ``None`` for external packages not present
    in the project.
    """
    module = import_info.module
    segments = module.split(".")
    target_path = str(PurePosixPath(*segments))
    return _try_python_paths(target_path, file_index)

def _try_python_paths(base_path: str, file_index: dict[str, str]) -> str | None:
    """Try common Python file resolution patterns for *base_path*.

    Checks in order:
    1. ``base_path.py`` (direct module file)
    2. ``base_path/__init__.py`` (package directory)
    """
    candidates = [
        f"{base_path}.py",
        f"{base_path}/__init__.py",
    ]
    for candidate in candidates:
        if candidate in file_index:
            return file_index[candidate]
    return None

def _resolve_js_ts(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a JavaScript/TypeScript import to a file node ID.

    Relative imports (starting with ``./`` or ``../``) are resolved against
    the importing file's directory.  Bare specifiers (e.g. ``'express'``)
    are treated as external and return ``None``.
    """
    import posixpath

    module = import_info.module

    if not module.startswith("."):
        return None

    base = str(PurePosixPath(importing_file).parent)
    resolved_str = posixpath.normpath(posixpath.join(base, module))

    return _try_js_ts_paths(resolved_str, file_index)

def _try_js_ts_paths(base_path: str, file_index: dict[str, str]) -> str | None:
    """Try common JS/TS file resolution patterns for *base_path*.

    Checks in order:
    1. ``base_path`` as-is (already has extension)
    2. ``base_path`` + each known extension (.ts, .js, .tsx, .jsx)
    3. ``base_path/index`` + each known extension
    """
    # 1. Exact match (import already includes extension).
    if base_path in file_index:
        return file_index[base_path]

    # 2. Try appending extensions.
    for ext in _JS_TS_EXTENSIONS:
        candidate = f"{base_path}{ext}"
        if candidate in file_index:
            return file_index[candidate]

    # 3. Try as directory with index file.
    for ext in _JS_TS_EXTENSIONS:
        candidate = f"{base_path}/index{ext}"
        if candidate in file_index:
            return file_index[candidate]

    return None


def _resolve_php(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    r"""Resolve a PHP import to a file node ID.

    Handles two forms:

    1. **Relative includes** (``require_once __DIR__ . '/helpers/utils.php'``)
       resolved relative to the importing file's directory.
    2. **Namespace ``use`` statements** (``use App\Services\UserService``)
       resolved via PSR-4-style path conversion.
    """
    module = import_info.module
    if not module:
        return None

    # Relative includes (require_once, include, etc.)
    if import_info.is_relative:
        return _resolve_php_include(importing_file, module, file_index)

    # Namespace-based use statements (PSR-4)
    # Convert namespace separators to path separators
    ns_path = module.replace("\\", "/")

    # Try the namespace path directly (e.g. App/Models/User.php)
    candidates = [
        f"{ns_path}.php",
    ]

    # Common PSR-4 prefix stripping: App\ -> app/, src/, lib/
    parts = ns_path.split("/")
    if len(parts) > 1:
        rest = "/".join(parts[1:])
        for prefix in ("app", "src", "lib"):
            candidates.append(f"{prefix}/{rest}.php")
        # Also try with lowercase first segment
        candidates.append(f"{parts[0].lower()}/{rest}.php")

    for candidate in candidates:
        if candidate in file_index:
            return file_index[candidate]

    return None


def _resolve_php_include(
    importing_file: str,
    module: str,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a PHP ``require_once`` / ``include`` to a file node ID.

    The *module* is a relative file path (e.g. ``helpers/utils.php`` or
    ``../config/config.php``) extracted from the include statement.
    Resolution is relative to the importing file's directory.
    """
    import posixpath

    base = str(PurePosixPath(importing_file).parent)
    joined = posixpath.join(base, module)
    resolved = posixpath.normpath(joined)

    if resolved in file_index:
        return file_index[resolved]

    return None


def _resolve_html(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve an HTML ``<script src>`` or ``<link href>`` to a file node ID.

    Treats relative script sources like JS/TS relative imports.
    Absolute URLs and CDN references are treated as external.
    Site-root paths (``/assets/...``) are converted to repo-relative paths.
    Query strings (``?v=2.0``) are stripped before resolution.
    """
    import posixpath

    module = import_info.module
    if not module:
        return None

    # Skip absolute URLs (CDN, protocol-relative)
    if module.startswith(("http://", "https://", "//")):
        return None

    # Strip query strings (e.g. "app.js?v=2.0" -> "app.js")
    if "?" in module:
        module = module.split("?")[0]

    # Site-root absolute paths (e.g. "/assets/vendor/react.js" -> "assets/vendor/react.js")
    if module.startswith("/"):
        module = module.lstrip("/")
        # For absolute paths, resolve against repo root (not importing file's dir)
        resolved_str = module
    else:
        # Resolve relative to the importing HTML file's directory
        base = str(PurePosixPath(importing_file).parent)
        resolved_str = posixpath.normpath(posixpath.join(base, module))

    # Try exact match first
    if resolved_str in file_index:
        return file_index[resolved_str]

    # Try with JS/TS extensions if no extension was specified
    if not PurePosixPath(resolved_str).suffix:
        return _try_js_ts_paths(resolved_str, file_index)

    return None
