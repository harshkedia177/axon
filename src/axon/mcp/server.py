"""MCP server for Axon — exposes code intelligence tools over stdio and HTTP.

Registers fifteen tools and three resources that give AI agents and MCP clients
access to the Axon knowledge graph.  The server lazily initialises a
:class:`KuzuBackend` from the ``.axon/kuzu`` directory in the current
working directory.

Usage::

    # MCP server only
    axon mcp

    # MCP server with live file watching (recommended)
    axon serve --watch
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from mcp.server import Server
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Resource, TextContent, Tool

from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.resources import get_dead_code_list, get_overview, get_schema
from axon.mcp.tools import (
    MAX_TRAVERSE_DEPTH,
    handle_call_path,
    handle_communities,
    handle_context,
    handle_coupling,
    handle_cycles,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_explain,
    handle_file_context,
    handle_impact,
    handle_list_repos,
    handle_query,
    handle_review_risk,
    handle_test_impact,
)

logger = logging.getLogger(__name__)

server = Server("axon")

_storage: KuzuBackend | None = None
_lock: asyncio.Lock | None = None

_db_path: Path | None = None


def _resolve_db_path() -> Path:
    global _db_path  # noqa: PLW0603
    if _db_path is None:
        _db_path = Path.cwd() / ".axon" / "kuzu"
    return _db_path


def set_storage(storage: KuzuBackend) -> None:
    """Inject a pre-initialised storage backend (e.g. from ``axon serve --watch``)."""
    global _storage  # noqa: PLW0603
    _storage = storage


def set_lock(lock: asyncio.Lock) -> None:
    """Inject a shared lock for coordinating storage access with the file watcher."""
    global _lock  # noqa: PLW0603
    _lock = lock


def set_db_path(path: Path) -> None:
    """Inject a custom database path for standalone MCP server mode.

    Must be called before the server handles any tool requests (i.e., before
    entering the event loop in ``main()``).
    """
    global _db_path  # noqa: PLW0603
    _db_path = path


@contextmanager
def _open_storage() -> Iterator[KuzuBackend]:
    """Open a short-lived read-only connection for a single tool/resource call.

    Used when no persistent storage was injected (read-only fallback mode).
    Each call gets a fresh connection that sees the latest on-disk data and
    releases the file lock immediately after the query completes.
    """
    db_path = _resolve_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"No .axon/kuzu directory in {db_path.parent.parent}")
    storage = KuzuBackend()
    storage.initialize(db_path, read_only=True, max_retries=3, retry_delay=0.3)
    try:
        yield storage
    finally:
        storage.close()


async def _with_storage(fn: Callable[[KuzuBackend], str]) -> str:
    """Run *fn* against the appropriate storage backend.

    Uses the injected persistent backend when available (with optional
    async lock), otherwise opens a short-lived read-only connection.
    """
    if _storage is not None:
        if _lock is not None:
            async with _lock:
                return await asyncio.to_thread(fn, _storage)
        return await asyncio.to_thread(fn, _storage)

    def _run() -> str:
        with _open_storage() as st:
            return fn(st)

    return await asyncio.to_thread(_run)


TOOLS: list[Tool] = [
    Tool(
        name="axon_list_repos",
        description="List all indexed repositories with their stats.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_query",
        description=(
            "Search the knowledge graph using hybrid (keyword + vector) search. "
            "Returns ranked symbols matching the query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="axon_context",
        description=(
            "Get a 360-degree view of a symbol: callers, callees, type references, "
            "and community membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to look up.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_impact",
        description=(
            "Blast radius analysis: find all symbols affected by changing a given symbol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to analyse.",
                },
                "depth": {
                    "type": "integer",
                    "description": (
                        f"Maximum traversal depth (default 3, max {MAX_TRAVERSE_DEPTH})."
                    ),
                    "default": 3,
                    "minimum": 1,
                    "maximum": MAX_TRAVERSE_DEPTH,
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_dead_code",
        description="List all symbols detected as dead (unreachable) code.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_detect_changes",
        description=(
            "Parse a git diff and map changed files/lines to affected symbols "
            "in the knowledge graph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
            },
            "required": ["diff"],
        },
    ),
    Tool(
        name="axon_cypher",
        description="Execute a raw Cypher query against the knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query string.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="axon_coupling",
        description=(
            "Show files temporally coupled with a given file. "
            "Reveals hidden dependencies from git co-change patterns."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to analyze coupling for.",
                },
                "min_strength": {
                    "type": "number",
                    "description": "Minimum coupling strength threshold (default 0.3).",
                    "default": 0.3,
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="axon_communities",
        description=(
            "List detected code communities (Leiden clusters) or drill into "
            "a specific community to see its members."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "community": {
                    "type": "string",
                    "description": "Optional community name to drill into. Omit to list all.",
                },
            },
        },
    ),
    Tool(
        name="axon_explain",
        description=(
            "Get a narrative explanation of a symbol: its role, community, "
            "process flows, and relationships summarized for onboarding."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to explain.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_review_risk",
        description=(
            "PR risk assessment: analyzes a git diff to find affected symbols, "
            "missing co-change files, community boundary crossings, and "
            "downstream blast radius. Returns a risk score."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
            },
            "required": ["diff"],
        },
    ),
    Tool(
        name="axon_call_path",
        description=(
            "Find the shortest call chain between two symbols. "
            "Uses BFS over CALLS edges."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "from_symbol": {
                    "type": "string",
                    "description": "Name of the source symbol.",
                },
                "to_symbol": {
                    "type": "string",
                    "description": "Name of the target symbol.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": f"Maximum hops (default 10, max {MAX_TRAVERSE_DEPTH}).",
                    "default": 10,
                    "minimum": 1,
                    "maximum": MAX_TRAVERSE_DEPTH,
                },
            },
            "required": ["from_symbol", "to_symbol"],
        },
    ),
    Tool(
        name="axon_file_context",
        description=(
            "Get comprehensive context for a file: symbols, imports, "
            "coupling, dead code, and community membership in one call."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to analyze.",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="axon_test_impact",
        description=(
            "Find tests likely affected by code changes. Accepts a git diff "
            "or symbol names, traces callers to find test files."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbol names to check.",
                },
            },
        },
    ),
    Tool(
        name="axon_cycles",
        description=(
            "Detect circular dependencies using strongly connected "
            "component analysis. Returns cycle groups sorted by size."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "min_size": {
                    "type": "integer",
                    "description": "Minimum cycle size to report (default 2).",
                    "default": 2,
                    "minimum": 2,
                },
            },
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available Axon tools."""
    return TOOLS


def _dispatch_tool(name: str, arguments: dict, storage: KuzuBackend) -> str:
    if name == "axon_list_repos":
        return handle_list_repos()
    elif name == "axon_query":
        return handle_query(storage, arguments.get("query", ""), limit=arguments.get("limit", 20))
    elif name == "axon_context":
        return handle_context(storage, arguments.get("symbol", ""))
    elif name == "axon_impact":
        return handle_impact(storage, arguments.get("symbol", ""), depth=arguments.get("depth", 3))
    elif name == "axon_dead_code":
        return handle_dead_code(storage)
    elif name == "axon_detect_changes":
        return handle_detect_changes(storage, arguments.get("diff", ""))
    elif name == "axon_cypher":
        return handle_cypher(storage, arguments.get("query", ""))
    elif name == "axon_coupling":
        return handle_coupling(
            storage, arguments.get("file_path", ""),
            min_strength=arguments.get("min_strength", 0.3),
        )
    elif name == "axon_communities":
        return handle_communities(storage, community=arguments.get("community"))
    elif name == "axon_explain":
        return handle_explain(storage, arguments.get("symbol", ""))
    elif name == "axon_review_risk":
        return handle_review_risk(storage, arguments.get("diff", ""))
    elif name == "axon_call_path":
        return handle_call_path(
            storage,
            arguments.get("from_symbol", ""),
            arguments.get("to_symbol", ""),
            max_depth=arguments.get("max_depth", 10),
        )
    elif name == "axon_file_context":
        return handle_file_context(storage, arguments.get("file_path", ""))
    elif name == "axon_test_impact":
        return handle_test_impact(
            storage,
            diff=arguments.get("diff", ""),
            symbols=arguments.get("symbols"),
        )
    elif name == "axon_cycles":
        return handle_cycles(
            storage, min_size=arguments.get("min_size", 2),
        )
    else:
        return f"Unknown tool: {name}"


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    try:
        result = await _with_storage(lambda st: _dispatch_tool(name, arguments, st))
    except Exception as exc:
        logger.exception("Tool %s raised an unhandled exception", name)
        result = f"Internal error: {exc}"

    return [TextContent(type="text", text=result)]


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return the list of available Axon resources."""
    return [
        Resource(
            uri="axon://overview",
            name="Codebase Overview",
            description="High-level statistics about the indexed codebase.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://dead-code",
            name="Dead Code Report",
            description="List of all symbols flagged as unreachable.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://schema",
            name="Graph Schema",
            description="Description of the Axon knowledge graph schema.",
            mimeType="text/plain",
        ),
    ]


def _dispatch_resource(uri_str: str, storage: KuzuBackend) -> str:
    if uri_str == "axon://overview":
        return get_overview(storage)
    if uri_str == "axon://dead-code":
        return get_dead_code_list(storage)
    if uri_str == "axon://schema":
        return get_schema()
    return f"Unknown resource: {uri_str}"


@server.read_resource()
async def read_resource(uri) -> str:
    """Read the contents of an Axon resource."""
    uri_str = str(uri)
    return await _with_storage(lambda st: _dispatch_resource(uri_str, st))


async def main() -> None:
    """Run the Axon MCP server over stdio transport."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def create_streamable_http_app() -> tuple[StreamableHTTPSessionManager, StreamableHTTPASGIApp]:
    """Create a streamable HTTP transport for the existing MCP server."""
    session_manager = StreamableHTTPSessionManager(app=server)
    return session_manager, StreamableHTTPASGIApp(session_manager)


if __name__ == "__main__":
    asyncio.run(main())
