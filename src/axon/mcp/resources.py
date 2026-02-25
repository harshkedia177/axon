"""MCP resource handlers for Axon.

Provides helper functions that generate formatted text for MCP resources.
These are read-only snapshots of graph state, exposed as text resources
that MCP clients can fetch.
"""

from __future__ import annotations

from axon.core.storage.base import StorageBackend


def get_overview(storage: StorageBackend) -> str:
    """Generate a high-level overview of the indexed codebase.

    Queries the storage backend for aggregate statistics and returns
    a human-readable summary.

    Args:
        storage: The storage backend.

    Returns:
        Formatted overview including node counts, file counts, and
        relationship statistics.
    """
    lines = ["Axon Codebase Overview", "=" * 40, ""]

    try:
        rows = storage.execute_read_query(
            "MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC"
        )
        if rows:
            lines.append("Node counts by type:")
            total = 0
            for row in rows:
                label = row[0] if row else "Unknown"
                count = row[1] if len(row) > 1 else 0
                lines.append(f"  {label}: {count}")
                total += count
            lines.append(f"  Total: {total}")
        else:
            lines.append("No nodes indexed yet.")
    except Exception:
        lines.append("Could not retrieve node counts.")

    lines.append("")

    try:
        rows = storage.execute_read_query(
            "MATCH ()-[r]->() RETURN r.rel_type, count(r) ORDER BY count(r) DESC"
        )
        if rows:
            lines.append("Relationship counts by type:")
            total = 0
            for row in rows:
                rel_type = row[0] if row else "Unknown"
                count = row[1] if len(row) > 1 else 0
                lines.append(f"  {rel_type}: {count}")
                total += count
            lines.append(f"  Total: {total}")
        else:
            lines.append("No relationships indexed yet.")
    except Exception:
        lines.append("Could not retrieve relationship counts.")

    return "\n".join(lines)

def get_dead_code_list(storage: StorageBackend) -> str:
    """Generate a formatted list of all dead code in the codebase.

    Args:
        storage: The storage backend.

    Returns:
        Formatted list of symbols flagged as dead code.
    """
    try:
        rows = storage.execute_read_query(
            "MATCH (n) WHERE n.is_dead = true "
            "RETURN n.name, n.file_path, n.start_line ORDER BY n.file_path"
        )
    except Exception:
        return "Could not retrieve dead code list."

    if not rows:
        return "No dead code detected. Codebase looks clean."

    lines = [f"Dead Code Report ({len(rows)} symbols)", "-" * 40, ""]
    current_file = ""
    for row in rows:
        name = row[0] if row else "?"
        file_path = row[1] if len(row) > 1 else "?"
        start_line = row[2] if len(row) > 2 else "?"
        if file_path != current_file:
            if current_file:
                lines.append("")
            lines.append(f"  {file_path}:")
            current_file = file_path
        lines.append(f"    - {name} (line {start_line})")

    return "\n".join(lines)

def get_schema() -> str:
    """Return a static description of the Axon knowledge graph schema.

    This does not require a storage connection since the schema is fixed.

    Returns:
        Human-readable description of node labels, relationship types,
        and key properties.
    """
    return """Axon Knowledge Graph Schema
========================================

Node Labels:
  - File       : Source file in the repository
  - Folder     : Directory in the repository
  - Function   : Top-level function definition
  - Class      : Class definition
  - Method     : Method within a class
  - Interface  : Interface / protocol definition
  - TypeAlias  : Type alias definition
  - Enum       : Enumeration definition
  - Community  : Detected community cluster (via Leiden algorithm)
  - Process    : Business process / workflow

Common Node Properties:
  id, name, file_path, start_line, end_line, content,
  signature, language, class_name, is_dead, is_entry_point, is_exported

Relationship Types:
  - CONTAINS      : Folder/File contains a symbol
  - DEFINES       : File defines a symbol
  - CALLS         : Symbol calls another symbol
  - IMPORTS       : File/symbol imports another
  - EXTENDS       : Class extends another class
  - IMPLEMENTS    : Class implements an interface
  - MEMBER_OF     : Symbol belongs to a community
  - STEP_IN_PROCESS : Symbol is a step in a process
  - USES_TYPE     : Symbol references a type
  - EXPORTS       : Module exports a symbol
  - COUPLED_WITH  : Temporal coupling between symbols

Relationship Properties:
  rel_type, confidence, role, step_number, strength, co_changes, symbols

ID Format:
  {label}:{file_path}:{symbol_name}
  Example: function:src/auth.py:validate_user
"""
