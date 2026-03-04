"""Tests for the process / execution flow detection phase (Phase 9)."""

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
from axon.core.ingestion.processes import (
    deduplicate_flows,
    find_entry_points,
    generate_process_label,
    process_processes,
    trace_flow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_function(
    graph: KnowledgeGraph,
    name: str,
    file_path: str = "src/app.py",
    *,
    content: str = "",
    language: str = "python",
    is_exported: bool = False,
) -> GraphNode:
    """Add a FUNCTION node and return it."""
    node_id = generate_id(NodeLabel.FUNCTION, file_path, name)
    node = GraphNode(
        id=node_id,
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        content=content,
        language=language,
        is_exported=is_exported,
    )
    graph.add_node(node)
    return node


def _add_call(
    graph: KnowledgeGraph,
    source: GraphNode,
    target: GraphNode,
    confidence: float = 1.0,
) -> None:
    """Add a CALLS relationship between two nodes."""
    rel_id = f"calls:{source.id}->{target.id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source.id,
            target=target.id,
            properties={"confidence": confidence},
        )
    )


def _add_member_of(
    graph: KnowledgeGraph,
    node: GraphNode,
    community_id: str,
) -> None:
    """Add a MEMBER_OF relationship from *node* to a community."""
    rel_id = f"member_of:{node.id}->{community_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.MEMBER_OF,
            source=node.id,
            target=community_id,
        )
    )


# ---------------------------------------------------------------------------
# Fixture: call graph
#
#   main() --> validate() --> hash_password()
#                         \-> query_db() --> format_result()
#
#   orphan_func() <-- (has incoming call from some_caller)
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the specification.

    - main() calls validate()
    - validate() calls hash_password() and query_db()
    - query_db() calls format_result()
    - orphan_func() has an incoming call (so it is NOT an entry point)
    """
    g = KnowledgeGraph()

    main = _add_function(g, "main")
    validate = _add_function(g, "validate")
    hash_password = _add_function(g, "hash_password")
    query_db = _add_function(g, "query_db")
    format_result = _add_function(g, "format_result")
    orphan_func = _add_function(g, "orphan_func")

    # Also add a caller for orphan_func so it has an incoming CALLS edge.
    some_caller = _add_function(g, "some_caller")

    _add_call(g, main, validate)
    _add_call(g, validate, hash_password)
    _add_call(g, validate, query_db)
    _add_call(g, query_db, format_result)
    _add_call(g, some_caller, orphan_func)

    return g


# ---------------------------------------------------------------------------
# 1. test_find_entry_points
# ---------------------------------------------------------------------------


class TestFindEntryPoints:
    """Entry points are functions with no incoming CALLS edges."""

    def test_find_entry_points(self, graph: KnowledgeGraph) -> None:
        """main is identified as entry point; orphan_func is NOT."""
        entry_points = find_entry_points(graph)
        ep_names = {n.name for n in entry_points}

        # main has no incoming CALLS -> entry point.
        assert "main" in ep_names
        # orphan_func HAS an incoming CALLS edge -> not an entry point
        # (unless matched by framework pattern, which it does not).
        assert "orphan_func" not in ep_names

    def test_entry_point_flag_set(self, graph: KnowledgeGraph) -> None:
        """is_entry_point is set to True on detected entry points."""
        entry_points = find_entry_points(graph)
        for ep in entry_points:
            assert ep.is_entry_point is True


# ---------------------------------------------------------------------------
# 2. test_find_entry_points_framework
# ---------------------------------------------------------------------------


class TestFindEntryPointsFramework:
    """Framework patterns are recognised as entry points."""

    def test_test_function_is_entry_point(self) -> None:
        """A function named test_something is detected as entry point."""
        g = KnowledgeGraph()
        test_fn = _add_function(g, "test_something", language="python")

        # Give it an incoming call so *only* the framework pattern triggers.
        caller = _add_function(g, "runner")
        _add_call(g, caller, test_fn)

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "test_something" in ep_names

    def test_decorator_pattern_entry_point(self) -> None:
        """A function with @app.route in content is an entry point."""
        g = KnowledgeGraph()
        _add_function(
            g,
            "index",
            content='@app.route("/")\ndef index():\n    pass',
            language="python",
        )

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "index" in ep_names

    def test_ts_handler_is_entry_point(self) -> None:
        """A TypeScript function named handler is an entry point."""
        g = KnowledgeGraph()
        _add_function(
            g,
            "handler",
            file_path="src/api.ts",
            language="typescript",
        )

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "handler" in ep_names


# ---------------------------------------------------------------------------
# 3. test_trace_flow
# ---------------------------------------------------------------------------


class TestTraceFlow:
    """BFS traces the correct path from an entry point."""

    def test_trace_flow(self, graph: KnowledgeGraph) -> None:
        """Tracing from main covers the full call chain."""
        main_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "main")
        main_node = graph.get_node(main_id)
        assert main_node is not None

        flow = trace_flow(main_node, graph)
        flow_names = [n.name for n in flow]

        # BFS from main: main -> validate -> {hash_password, query_db} -> format_result
        assert flow_names[0] == "main"
        assert "validate" in flow_names
        assert "hash_password" in flow_names
        assert "query_db" in flow_names
        assert "format_result" in flow_names
        assert len(flow) == 5

    def test_trace_flow_no_cycles(self, graph: KnowledgeGraph) -> None:
        """Visited tracking prevents infinite loops in cyclic graphs."""
        g = KnowledgeGraph()
        a = _add_function(g, "a")
        b = _add_function(g, "b")
        _add_call(g, a, b)
        _add_call(g, b, a)  # cycle

        flow = trace_flow(a, g)
        assert len(flow) == 2  # a, b -- no revisit


# ---------------------------------------------------------------------------
# 4. test_trace_flow_max_depth
# ---------------------------------------------------------------------------


class TestTraceFlowMaxDepth:
    """Depth limit is respected."""

    def test_trace_flow_max_depth(self, graph: KnowledgeGraph) -> None:
        """With max_depth=1, only the direct callees are included."""
        main_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "main")
        main_node = graph.get_node(main_id)
        assert main_node is not None

        flow = trace_flow(main_node, graph, max_depth=1)
        flow_names = [n.name for n in flow]

        # main -> validate (depth 1), but hash_password/query_db at depth 2 are cut off.
        assert "main" in flow_names
        assert "validate" in flow_names
        # Depth-2 nodes should NOT appear.
        assert "hash_password" not in flow_names
        assert "query_db" not in flow_names


# ---------------------------------------------------------------------------
# 5. test_generate_process_label
# ---------------------------------------------------------------------------


class TestGenerateProcessLabel:
    """Process labels are formatted correctly."""

    def test_generate_process_label(self) -> None:
        """Multi-step label uses arrow notation with max 4 steps."""
        nodes = [
            GraphNode(id=f"n{i}", label=NodeLabel.FUNCTION, name=name)
            for i, name in enumerate(
                ["main", "validate", "hash_password", "query_db", "format_result"]
            )
        ]
        label = generate_process_label(nodes)
        # Max 4 steps in the label.
        assert label == "main \u2192 validate \u2192 hash_password \u2192 query_db"

    def test_generate_process_label_single(self) -> None:
        """Single-step label is just the function name."""
        nodes = [GraphNode(id="n0", label=NodeLabel.FUNCTION, name="main")]
        label = generate_process_label(nodes)
        assert label == "main"

    def test_generate_process_label_empty(self) -> None:
        """Empty input gives empty string."""
        assert generate_process_label([]) == ""


# ---------------------------------------------------------------------------
# 6. test_deduplicate_flows
# ---------------------------------------------------------------------------


class TestDeduplicateFlows:
    """Similar flows are merged by keeping the longer one."""

    def test_deduplicate_flows(self) -> None:
        """A short flow that overlaps >70% with a longer flow is discarded."""
        # Create nodes.
        a = GraphNode(id="a", label=NodeLabel.FUNCTION, name="a")
        b = GraphNode(id="b", label=NodeLabel.FUNCTION, name="b")
        c = GraphNode(id="c", label=NodeLabel.FUNCTION, name="c")
        d = GraphNode(id="d", label=NodeLabel.FUNCTION, name="d")

        long_flow = [a, b, c, d]
        short_flow = [a, b, c]  # 100% overlap with long_flow (3/3)

        result = deduplicate_flows([short_flow, long_flow])
        assert len(result) == 1
        assert len(result[0]) == 4  # Kept the longer flow.

    def test_deduplicate_keeps_distinct(self) -> None:
        """Flows with low overlap are both kept."""
        a = GraphNode(id="a", label=NodeLabel.FUNCTION, name="a")
        b = GraphNode(id="b", label=NodeLabel.FUNCTION, name="b")
        c = GraphNode(id="c", label=NodeLabel.FUNCTION, name="c")
        d = GraphNode(id="d", label=NodeLabel.FUNCTION, name="d")

        flow1 = [a, b]
        flow2 = [c, d]

        result = deduplicate_flows([flow1, flow2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 7. test_process_processes_creates_nodes
# ---------------------------------------------------------------------------


class TestProcessProcessesCreatesNodes:
    """process_processes creates Process nodes in the graph."""

    def test_process_processes_creates_nodes(self, graph: KnowledgeGraph) -> None:
        process_processes(graph)

        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        assert len(process_nodes) > 0

        # Each Process node has a name and step_count property.
        for pn in process_nodes:
            assert pn.name != ""
            assert pn.properties["step_count"] > 1


# ---------------------------------------------------------------------------
# 8. test_process_processes_creates_steps
# ---------------------------------------------------------------------------


class TestProcessProcessesCreatesSteps:
    """STEP_IN_PROCESS relationships are created with step numbers."""

    def test_process_processes_creates_steps(self, graph: KnowledgeGraph) -> None:
        process_processes(graph)

        step_rels = graph.get_relationships_by_type(RelType.STEP_IN_PROCESS)
        assert len(step_rels) > 0

        # All step relationships should have a step_number property.
        for rel in step_rels:
            assert "step_number" in rel.properties
            assert isinstance(rel.properties["step_number"], int)

        # Verify step numbers start at 0 for each process.
        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        for pn in process_nodes:
            incoming = graph.get_incoming(pn.id, RelType.STEP_IN_PROCESS)
            step_numbers = sorted(r.properties["step_number"] for r in incoming)
            assert step_numbers[0] == 0
            assert step_numbers == list(range(len(step_numbers)))


# ---------------------------------------------------------------------------
# 9. test_process_processes_returns_count
# ---------------------------------------------------------------------------


class TestProcessProcessesReturnsCount:
    """process_processes returns the correct count of processes created."""

    def test_process_processes_returns_count(self, graph: KnowledgeGraph) -> None:
        count = process_processes(graph)

        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        assert count == len(process_nodes)
        assert count > 0
