"""Phase 7: Type analysis for Axon.

Takes FileParseData from the parser phase and resolves type annotation
references to their corresponding Class, Interface, or TypeAlias nodes,
creating USES_TYPE relationships from Function/Method nodes to the resolved
type nodes.
"""

from __future__ import annotations

import logging

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import (
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)

logger = logging.getLogger(__name__)

_TYPE_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.CLASS,
    NodeLabel.INTERFACE,
    NodeLabel.TYPE_ALIAS,
)

_CONTAINER_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
)

def _resolve_type(
    type_name: str,
    file_path: str,
    type_index: dict[str, list[str]],
    graph: KnowledgeGraph,
) -> str | None:
    """Resolve a type name to a target node ID.

    Resolution strategy (tried in order):

    1. **Same-file match** -- the type is defined in the same file as the
       reference.
    2. **Global match** -- any type with this name anywhere in the codebase.
       If multiple matches exist the first one is returned.

    Args:
        type_name: The referenced type name (e.g. ``"User"``).
        file_path: Path to the file containing the type reference.
        type_index: Mapping from type names to node IDs built by
            :func:`build_type_index`.
        graph: The knowledge graph.

    Returns:
        The node ID of the resolved type, or ``None`` if unresolved.
    """
    candidate_ids = type_index.get(type_name, [])
    if not candidate_ids:
        return None

    # 1. Same-file match.
    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path == file_path:
            return nid

    # 2. Global match -- return the first candidate.
    return candidate_ids[0]

def process_types(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> None:
    """Resolve type references and create USES_TYPE relationships in the graph.

    For each type reference in the parse data:

    1. Determine which Function/Method in the file *contains* the reference
       (by line number range).
    2. Resolve the type name to a Class, Interface, or TypeAlias node.
    3. Create a USES_TYPE relationship from the containing symbol to the
       resolved type node, with a ``role`` property set to the kind of
       reference (``"param"``, ``"return"``, or ``"variable"``).

    Skips type references where:
    - The containing symbol cannot be determined.
    - The type name cannot be resolved (built-in or external).
    - A relationship with the same ID already exists (deduplication).

    Args:
        parse_data: File parse results from the parser phase.
        graph: The knowledge graph to populate with USES_TYPE relationships.
    """
    type_index = build_name_index(graph, _TYPE_LABELS)
    file_sym_index = build_file_symbol_index(graph, _CONTAINER_LABELS)
    seen: set[str] = set()

    for fpd in parse_data:
        for type_ref in fpd.parse_result.type_refs:
            source_id = find_containing_symbol(
                type_ref.line, fpd.file_path, file_sym_index
            )
            if source_id is None:
                logger.debug(
                    "No containing symbol for type ref %s at line %d in %s",
                    type_ref.name,
                    type_ref.line,
                    fpd.file_path,
                )
                continue

            target_id = _resolve_type(
                type_ref.name, fpd.file_path, type_index, graph
            )
            if target_id is None:
                continue

            role = type_ref.kind
            rel_id = f"uses_type:{source_id}->{target_id}:{role}"
            if rel_id in seen:
                continue
            seen.add(rel_id)

            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.USES_TYPE,
                    source=source_id,
                    target=target_id,
                    properties={"role": role},
                )
            )
