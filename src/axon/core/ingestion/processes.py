"""Phase 9: Process / execution flow detection for Axon.

Detects execution flows by finding entry points and tracing call chains
via BFS.  Creates Process nodes and STEP_IN_PROCESS relationships that
represent end-to-end execution paths through the codebase.
"""

from __future__ import annotations

import logging
from collections import deque

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)

logger = logging.getLogger(__name__)

_CALLABLE_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
)

_MAX_FLOW_SIZE = 25

_PYTHON_DECORATOR_PATTERNS: tuple[str, ...] = (
    "@app.route",
    "@router",
    "@click.command",
)

# PHP file-path segments that indicate an entry point (API endpoint, cron job,
# webhook receiver, controller, or route handler).
_PHP_ENTRY_PATH_SEGMENTS: tuple[str, ...] = (
    "/api/",
    "/routes/",
    "/controllers/",
    "/Controllers/",
    "/cron/",
    "/webhooks/",
    "/public/",
)

# PHP file-name patterns for entry points.
_PHP_ENTRY_FILE_PATTERNS: tuple[str, ...] = (
    "-webhook.php",
    "_webhook.php",
)

# PHP content patterns that signal a file handles HTTP requests directly.
_PHP_ENTRY_CONTENT_PATTERNS: tuple[str, ...] = (
    "$_GET[",
    "$_POST[",
    "$_REQUEST[",
    "$_SERVER['REQUEST_METHOD']",
    '$_SERVER["REQUEST_METHOD"]',
)

def find_entry_points(graph: KnowledgeGraph) -> list[GraphNode]:
    """Find functions/methods that serve as execution entry points.

    A node is an entry point if it has NO incoming CALLS relationships,
    or if it matches a recognised framework pattern:

    - **Python**: functions starting with ``test_``, functions named
      ``main``, functions whose content contains decorator patterns like
      ``@app.route``, ``@router``, ``@click.command``.
    - **TypeScript**: functions named ``handler`` or ``middleware``,
      exported functions.

    Each identified entry point has its ``is_entry_point`` attribute set
    to ``True``.

    Args:
        graph: The knowledge graph to scan.

    Returns:
        A list of entry point :class:`GraphNode` instances.
    """
    entry_points: list[GraphNode] = []

    for label in _CALLABLE_LABELS:
        for node in graph.get_nodes_by_label(label):
            if _is_entry_point(node, graph):
                node.is_entry_point = True
                entry_points.append(node)

    return entry_points

def _is_entry_point(node: GraphNode, graph: KnowledgeGraph) -> bool:
    """Determine whether *node* qualifies as an entry point.

    Framework patterns always qualify.  For functions with no incoming calls,
    we require additional evidence (name heuristics, exported status) to avoid
    marking every utility function as an entry point in large codebases.
    """
    if _matches_framework_pattern(node):
        return True

    incoming_calls = graph.get_incoming(node.id, RelType.CALLS)
    if incoming_calls:
        return False

    if node.is_exported:
        return True

    if node.name in ("main", "cli", "run", "app", "handler", "entrypoint"):
        return True

    if node.label == NodeLabel.FUNCTION and node.file_path.endswith(
        ("__main__.py", "cli.py", "main.py", "app.py")
    ):
        return True

    # PHP: top-level functions in cron files are entry points.
    if node.label == NodeLabel.FUNCTION and node.file_path.endswith(".php"):
        fp = node.file_path
        if "/cron/" in fp or fp.startswith("cron/") or fp.rsplit("/", 1)[-1].startswith("cron-"):
            return True

    return False

def _matches_framework_pattern(node: GraphNode) -> bool:
    """Check whether *node* matches a known framework entry point pattern."""
    name = node.name
    language = node.language.lower() if node.language else ""
    content = node.content or ""

    if language in ("python", "py", "") or node.file_path.endswith(".py"):
        if name.startswith("test_"):
            return True
        if name == "main":
            return True
        for pattern in _PYTHON_DECORATOR_PATTERNS:
            if pattern in content:
                return True

    if language in ("typescript", "tsx", "ts", "") or node.file_path.endswith(
        (".ts", ".tsx")
    ):
        if name in ("handler", "middleware"):
            return True
        if node.is_exported:
            return True

    # PHP patterns
    if language in ("php", "") or node.file_path.endswith(".php"):
        # Controller classes/methods
        if name.endswith("Controller") or (
            node.class_name and node.class_name.endswith("Controller")
        ):
            return True
        # File-path heuristics: API endpoints, cron jobs, webhooks, etc.
        fp = node.file_path
        for seg in _PHP_ENTRY_PATH_SEGMENTS:
            # Check both "/api/" in path and path starting with "api/".
            if seg in fp or fp.startswith(seg.lstrip("/")):
                return True
        for pat in _PHP_ENTRY_FILE_PATTERNS:
            if fp.endswith(pat):
                return True
        # Content-based: direct HTTP request handling
        for pattern in _PHP_ENTRY_CONTENT_PATTERNS:
            if pattern in content:
                return True

    return False

def trace_flow(
    entry_point: GraphNode,
    graph: KnowledgeGraph,
    max_depth: int = 6,
    max_branching: int = 3,
) -> list[GraphNode]:
    """BFS from *entry_point* through CALLS edges.

    At each level, at most *max_branching* callees are followed (those
    with higher confidence on the CALLS edge are preferred).  Traversal
    stops after *max_depth* levels, when the flow reaches
    :data:`_MAX_FLOW_SIZE` nodes, or when no unvisited callees remain.

    Args:
        entry_point: The starting node for the flow.
        graph: The knowledge graph.
        max_depth: Maximum BFS depth.
        max_branching: Maximum callees to follow per node at each level.

    Returns:
        An ordered list of nodes in the flow, starting with *entry_point*.
    """
    visited: set[str] = {entry_point.id}
    result: list[GraphNode] = [entry_point]

    queue: deque[tuple[str, int]] = deque([(entry_point.id, 0)])

    while queue:
        if len(result) >= _MAX_FLOW_SIZE:
            break

        current_id, depth = queue.popleft()

        if depth >= max_depth:
            continue

        outgoing = graph.get_outgoing(current_id, RelType.CALLS)
        outgoing.sort(
            key=lambda r: r.properties.get("confidence", 0.0), reverse=True
        )

        count = 0
        for rel in outgoing:
            if count >= max_branching or len(result) >= _MAX_FLOW_SIZE:
                break
            target_id = rel.target
            if target_id in visited:
                continue
            target_node = graph.get_node(target_id)
            if target_node is None:
                continue

            visited.add(target_id)
            result.append(target_node)
            queue.append((target_id, depth + 1))
            count += 1

    return result

def generate_process_label(steps: list[GraphNode]) -> str:
    """Create a human-readable label from the flow steps.

    Format: ``"EntryName -> Step2 -> Step3"`` (max 4 steps in label).
    If only one step, returns just the function name.

    Args:
        steps: Ordered list of nodes in the flow.

    Returns:
        A string label for the process.
    """
    if not steps:
        return ""

    if len(steps) == 1:
        return steps[0].name

    names = [s.name for s in steps[:4]]
    return " \u2192 ".join(names)

def deduplicate_flows(flows: list[list[GraphNode]]) -> list[list[GraphNode]]:
    """Remove flows that are too similar to longer ones.

    Two flows are "similar" if they share > 70% of their nodes (by ID).
    When a pair is similar, the shorter flow is discarded.

    Args:
        flows: List of flows (each flow is a list of nodes).

    Returns:
        Deduplicated list of flows.
    """
    flows_sorted = sorted(flows, key=len, reverse=True)

    kept: list[list[GraphNode]] = []
    kept_sets: list[set[str]] = []

    for flow in flows_sorted:
        flow_ids = {n.id for n in flow}
        is_duplicate = False

        for kept_set in kept_sets:
            if not flow_ids or not kept_set:
                continue
            intersection = flow_ids & kept_set
            smaller_size = min(len(flow_ids), len(kept_set))
            overlap = len(intersection) / smaller_size
            if overlap > 0.5:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(flow)
            kept_sets.append(flow_ids)

    return kept

def _determine_kind(steps: list[GraphNode], graph: KnowledgeGraph) -> str:
    """Determine whether a flow is intra- or cross-community.

    Checks MEMBER_OF relationships for each step node. If all belong to
    the same community: ``"intra_community"``. If they span multiple:
    ``"cross_community"``. If no communities are assigned: ``"unknown"``.
    """
    communities: set[str] = set()
    has_any = False

    for step in steps:
        member_rels = graph.get_outgoing(step.id, RelType.MEMBER_OF)
        for rel in member_rels:
            has_any = True
            communities.add(rel.target)

    if not has_any:
        return "unknown"
    if len(communities) <= 1:
        return "intra_community"
    return "cross_community"

def process_processes(graph: KnowledgeGraph) -> int:
    """Detect execution flows and create Process nodes in the graph.

    Steps:
    1. Find all entry points.
    2. Trace a flow from each entry point.
    3. Deduplicate similar flows.
    4. Filter out trivial flows (single step only).
    5. Create a Process node and STEP_IN_PROCESS relationships for each flow.

    Args:
        graph: The knowledge graph to enrich.

    Returns:
        The number of Process nodes created.
    """
    entry_points = find_entry_points(graph)
    logger.debug("Found %d entry points", len(entry_points))

    flows: list[list[GraphNode]] = []
    for ep in entry_points:
        flow = trace_flow(ep, graph)
        flows.append(flow)

    flows = deduplicate_flows(flows)
    flows = [f for f in flows if len(f) > 1]

    count = 0
    for i, steps in enumerate(flows):
        process_id = generate_id(NodeLabel.PROCESS, f"process_{i}")
        label = generate_process_label(steps)
        kind = _determine_kind(steps, graph)

        process_node = GraphNode(
            id=process_id,
            label=NodeLabel.PROCESS,
            name=label,
            properties={"step_count": len(steps), "kind": kind},
        )
        graph.add_node(process_node)

        for step_number, step in enumerate(steps):
            rel_id = f"step:{step.id}->{process_id}:{step_number}"
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.STEP_IN_PROCESS,
                    source=step.id,
                    target=process_id,
                    properties={"step_number": step_number},
                )
            )

        count += 1

    logger.info("Created %d process nodes", count)
    return count
