"""Shared file-path heuristics used across ingestion phases."""

from __future__ import annotations


def is_test_file(file_path: str) -> bool:
    """Return True if *file_path* belongs to a test directory or file."""
    parts = file_path.replace("\\", "/").split("/")
    return (
        "tests" in parts
        or "test" in parts
        or any(p.startswith("test_") for p in parts)
        or file_path.endswith("conftest.py")
    )


def is_alembic_migration(file_path: str) -> bool:
    """Return True if *file_path* looks like an Alembic migration file."""
    parts = file_path.replace("\\", "/").split("/")
    return "versions" in parts or "migrations" in parts
