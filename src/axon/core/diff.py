"""Branch comparison for Axon.

Compares two code graphs structurally to find added, removed, and modified
nodes and relationships.  Uses git worktrees to avoid stashing or branch
switching in the user's working tree.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from axon.core.graph.model import GraphNode, GraphRelationship

logger = logging.getLogger(__name__)

@dataclass
class StructuralDiff:
    """Result of comparing two code graphs."""

    added_nodes: list[GraphNode] = field(default_factory=list)
    removed_nodes: list[GraphNode] = field(default_factory=list)
    modified_nodes: list[tuple[GraphNode, GraphNode]] = field(default_factory=list)
    added_relationships: list[GraphRelationship] = field(default_factory=list)
    removed_relationships: list[GraphRelationship] = field(default_factory=list)

# Fields checked to determine if a node was "modified".
_NODE_COMPARE_FIELDS = ("content", "signature", "start_line", "end_line")

def diff_graphs(
    base_nodes: dict[str, GraphNode],
    current_nodes: dict[str, GraphNode],
    base_rels: dict[str, GraphRelationship],
    current_rels: dict[str, GraphRelationship],
) -> StructuralDiff:
    """Diff two graph snapshots by node/relationship IDs.

    Nodes present only in *current_nodes* are added; only in *base_nodes* are
    removed.  Nodes with the same ID but different content/signature/lines are
    modified.  Relationships are compared by ID only (added/removed).

    Args:
        base_nodes: ``{node_id: GraphNode}`` from the base branch.
        current_nodes: ``{node_id: GraphNode}`` from the current branch.
        base_rels: ``{rel_id: GraphRelationship}`` from the base branch.
        current_rels: ``{rel_id: GraphRelationship}`` from the current branch.

    Returns:
        A :class:`StructuralDiff` with the comparison results.
    """
    result = StructuralDiff()

    base_ids = set(base_nodes)
    current_ids = set(current_nodes)

    for nid in current_ids - base_ids:
        result.added_nodes.append(current_nodes[nid])

    for nid in base_ids - current_ids:
        result.removed_nodes.append(base_nodes[nid])

    for nid in base_ids & current_ids:
        base_node = base_nodes[nid]
        current_node = current_nodes[nid]
        if _node_changed(base_node, current_node):
            result.modified_nodes.append((base_node, current_node))

    base_rel_ids = set(base_rels)
    current_rel_ids = set(current_rels)

    for rid in current_rel_ids - base_rel_ids:
        result.added_relationships.append(current_rels[rid])

    for rid in base_rel_ids - current_rel_ids:
        result.removed_relationships.append(base_rels[rid])

    return result

def _node_changed(base: GraphNode, current: GraphNode) -> bool:
    """Return True if the two nodes differ on any comparison field."""
    for attr in _NODE_COMPARE_FIELDS:
        if getattr(base, attr) != getattr(current, attr):
            return True
    return False

def diff_branches(
    repo_path: Path,
    branch_range: str,
) -> StructuralDiff:
    """Compare two branches structurally using git worktrees.

    *branch_range* should be ``"base..current"`` (e.g. ``"main..feature"``).
    If only one branch is given (no ``..``), it is treated as the base and
    the current working tree is used as the current branch.

    Steps:
        1. Parse branch range into base/current references.
        2. Create a temporary worktree for the base branch.
        3. Run the pipeline on both branches to build in-memory graphs.
        4. Diff the two graphs.
        5. Clean up the worktree.

    Args:
        repo_path: Root of the git repository.
        branch_range: Branch range string (e.g. ``"main..feature"``).

    Returns:
        A :class:`StructuralDiff` comparing the two branches.

    Raises:
        ValueError: If the branch range format is invalid.
        RuntimeError: If git operations fail.
    """
    from axon.core.ingestion.pipeline import build_graph

    if ".." in branch_range:
        parts = branch_range.split("..", 1)
        base_ref = parts[0].strip()
        current_ref = parts[1].strip() if parts[1].strip() else None
    else:
        base_ref = branch_range.strip()
        current_ref = None

    if not base_ref:
        raise ValueError(f"Invalid branch range: {branch_range!r}")

    # Build both graphs (in parallel when both need worktrees).
    if current_ref:
        with ThreadPoolExecutor(max_workers=2) as executor:
            base_future = executor.submit(_build_graph_for_ref, repo_path, base_ref)
            current_future = executor.submit(_build_graph_for_ref, repo_path, current_ref)
            base_graph = base_future.result()
            current_graph = current_future.result()
    else:
        current_graph = build_graph(repo_path)
        base_graph = _build_graph_for_ref(repo_path, base_ref)

    base_nodes = {n.id: n for n in base_graph.iter_nodes()}
    current_nodes = {n.id: n for n in current_graph.iter_nodes()}
    base_rels = {r.id: r for r in base_graph.iter_relationships()}
    current_rels = {r.id: r for r in current_graph.iter_relationships()}

    return diff_graphs(base_nodes, current_nodes, base_rels, current_rels)

def _build_graph_for_ref(repo_path: Path, ref: str) -> "KnowledgeGraph":
    """Build an in-memory graph for a git ref using a temporary worktree."""
    from axon.core.ingestion.pipeline import build_graph

    with tempfile.TemporaryDirectory(prefix="axon_diff_") as tmp_dir:
        worktree_path = Path(tmp_dir) / "worktree"

        try:
            subprocess.run(
                ["git", "worktree", "add", str(worktree_path), ref],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create worktree for ref '{ref}': {exc.stderr.strip()}"
            ) from exc

        try:
            graph = build_graph(worktree_path)
        finally:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree_path)],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                logger.warning("Failed to remove worktree at %s", worktree_path)

    return graph

def format_diff(diff: StructuralDiff) -> str:
    """Format a StructuralDiff as human-readable output.

    Args:
        diff: The structural diff to format.

    Returns:
        A multi-line string summarizing added, removed, and modified entities.
    """
    total_changes = (
        len(diff.added_nodes)
        + len(diff.removed_nodes)
        + len(diff.modified_nodes)
        + len(diff.added_relationships)
        + len(diff.removed_relationships)
    )

    if total_changes == 0:
        return "No structural differences found."

    lines: list[str] = []
    lines.append(f"Structural diff: {total_changes} changes")
    lines.append("")

    if diff.added_nodes:
        lines.append(f"Added nodes ({len(diff.added_nodes)}):")
        for node in sorted(diff.added_nodes, key=lambda n: n.id):
            label = node.label.value.title()
            lines.append(f"  + {node.name} ({label}) -- {node.file_path}")
        lines.append("")

    if diff.removed_nodes:
        lines.append(f"Removed nodes ({len(diff.removed_nodes)}):")
        for node in sorted(diff.removed_nodes, key=lambda n: n.id):
            label = node.label.value.title()
            lines.append(f"  - {node.name} ({label}) -- {node.file_path}")
        lines.append("")

    if diff.modified_nodes:
        lines.append(f"Modified nodes ({len(diff.modified_nodes)}):")
        for base_node, current_node in sorted(diff.modified_nodes, key=lambda p: p[0].id):
            label = current_node.label.value.title()
            lines.append(f"  ~ {current_node.name} ({label}) -- {current_node.file_path}")
        lines.append("")

    if diff.added_relationships:
        lines.append(f"Added relationships ({len(diff.added_relationships)}):")
        for rel in sorted(diff.added_relationships, key=lambda r: r.id):
            lines.append(f"  + [{rel.type.value}] {rel.source} -> {rel.target}")
        lines.append("")

    if diff.removed_relationships:
        lines.append(f"Removed relationships ({len(diff.removed_relationships)}):")
        for rel in sorted(diff.removed_relationships, key=lambda r: r.id):
            lines.append(f"  - [{rel.type.value}] {rel.source} -> {rel.target}")
        lines.append("")

    return "\n".join(lines).rstrip()
