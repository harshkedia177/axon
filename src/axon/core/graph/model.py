"""Knowledge graph data model for Axon.

Defines the core node and relationship types that represent code-level
entities (files, functions, classes, etc.) and the edges between them
(calls, imports, contains, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeLabel(Enum):
    """Labels for graph nodes representing code-level entities."""

    FILE = "file"
    FOLDER = "folder"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    COMMUNITY = "community"
    PROCESS = "process"

class RelType(Enum):
    """Relationship types connecting graph nodes."""

    CONTAINS = "contains"
    DEFINES = "defines"
    CALLS = "calls"
    IMPORTS = "imports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    MEMBER_OF = "member_of"
    STEP_IN_PROCESS = "step_in_process"
    USES_TYPE = "uses_type"
    EXPORTS = "exports"
    COUPLED_WITH = "coupled_with"

def generate_id(label: NodeLabel, file_path: str, symbol_name: str = "") -> str:
    """Produce a deterministic node ID.

    Format: ``{label.value}:{file_path}:{symbol_name}``

    Args:
        label: The node label enum member.
        file_path: Path to the file the symbol belongs to.
        symbol_name: Optional name of the symbol within the file.

    Returns:
        A colon-separated string suitable for use as a graph node ID.
    """
    return f"{label.value}:{file_path}:{symbol_name}"

@dataclass
class GraphNode:
    """A node in the knowledge graph representing a code entity.

    ``id``, ``label``, and ``name`` are required. All other fields carry
    sensible defaults so callers only need to supply what they know.
    """

    id: str
    label: NodeLabel
    name: str

    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    signature: str = ""
    language: str = ""
    class_name: str = ""

    is_dead: bool = False
    is_entry_point: bool = False
    is_exported: bool = False

    properties: dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphRelationship:
    """A directed edge in the knowledge graph.

    ``id``, ``type``, ``source``, and ``target`` are required.
    """

    id: str
    type: RelType
    source: str
    target: str

    properties: dict[str, Any] = field(default_factory=dict)
