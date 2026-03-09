"""Phase 9: Process / execution flow detection for Axon."""

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

# TypeScript files where exports are true entry points (index/entry/app files).
_TS_ENTRY_SUFFIXES: tuple[str, ...] = (
    "index.ts", "index.tsx", "index.js", "index.jsx",
    "main.ts", "main.tsx", "main.js",
    "app.ts", "app.tsx", "app.js",
    "server.ts", "server.js",
    "handler.ts", "handler.js",
    "route.ts", "route.tsx",
    "page.tsx", "page.ts",
    "layout.tsx", "layout.ts",
)


def _is_ts_entry_file(file_path: str) -> bool:
    return any(file_path.endswith(suffix) for suffix in _TS_ENTRY_SUFFIXES)

def find_entry_points(graph: KnowledgeGraph) -> list[GraphNode]:
    """Find functions/methods that serve as execution entry points."""
    entry_points: list[GraphNode] = []

    for label in _CALLABLE_LABELS:
        for node in graph.get_nodes_by_label(label):
            if _is_entry_point(node, graph):
                node.is_entry_point = True
                entry_points.append(node)

    return entry_points

def _is_entry_point(node: GraphNode, graph: KnowledgeGraph) -> bool:
    if _matches_framework_pattern(node):
        return True

    if graph.has_incoming(node.id, RelType.CALLS):
        return False

    if node.is_exported:
        return True

    if node.name in ("main", "cli", "run", "app", "handler", "entrypoint"):
        return True

    if node.label == NodeLabel.FUNCTION and node.file_path.endswith(
        ("__main__.py", "cli.py", "main.py", "app.py")
    ):
        return True

    return False

def _matches_framework_pattern(node: GraphNode) -> bool:
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

    if language in ("typescript", "ts", "") or node.file_path.endswith(
        (".ts", ".tsx")
    ):
        if name in ("handler", "middleware"):
            return True
        if node.is_exported and _is_ts_entry_file(node.file_path):
            return True

    return False


def trace_flow(
    entry_point: GraphNode,
    graph: KnowledgeGraph,
    max_depth: int = 6,
    max_branching: int = 3,
) -> list[GraphNode]:
    """BFS from entry_point through CALLS edges, up to max_depth levels."""
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
    """Create a human-readable label from the flow steps (max 4 names joined by →)."""
    if not steps:
        return ""

    if len(steps) == 1:
        return steps[0].name

    names = [s.name for s in steps[:4]]
    return " \u2192 ".join(names)

def deduplicate_flows(flows: list[list[GraphNode]]) -> list[list[GraphNode]]:
    """Remove flows that share >50% of nodes with a longer flow."""
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
    """Return "intra_community", "cross_community", or "unknown" for a flow."""
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
    """Detect execution flows and create Process nodes in the graph."""
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
