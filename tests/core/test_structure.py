"""Tests for the structure processor (Phase 2)."""

from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel, RelType, generate_id
from axon.core.ingestion.structure import process_structure
from axon.core.ingestion.walker import FileEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a fresh, empty KnowledgeGraph."""
    return KnowledgeGraph()


def _make_files(*paths: str, content: str = "", language: str = "python") -> list[FileEntry]:
    """Build a list of FileInfo entries from paths with shared defaults."""
    return [FileEntry(path=p, content=content, language=language) for p in paths]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreatesFileNodes:
    """test_creates_file_nodes — creates File nodes for each file."""

    def test_creates_file_nodes(self, graph: KnowledgeGraph) -> None:
        files = _make_files("src/auth/validate.py", "src/auth/crypto.py", "src/models/user.py")
        process_structure(files, graph)

        file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)
        assert len(file_nodes) == 3

        file_ids = {n.id for n in file_nodes}
        assert generate_id(NodeLabel.FILE, "src/auth/validate.py") in file_ids
        assert generate_id(NodeLabel.FILE, "src/auth/crypto.py") in file_ids
        assert generate_id(NodeLabel.FILE, "src/models/user.py") in file_ids


class TestCreatesFolderNodes:
    """test_creates_folder_nodes — creates Folder nodes for each unique directory."""

    def test_creates_folder_nodes(self, graph: KnowledgeGraph) -> None:
        files = _make_files("src/auth/validate.py", "src/models/user.py")
        process_structure(files, graph)

        folder_nodes = graph.get_nodes_by_label(NodeLabel.FOLDER)
        folder_paths = {n.file_path for n in folder_nodes}

        assert "src" in folder_paths
        assert "src/auth" in folder_paths
        assert "src/models" in folder_paths
        assert len(folder_nodes) == 3


class TestCreatesContainsRelationships:
    """test_creates_contains_relationships — folder contains files."""

    def test_creates_contains_relationships(self, graph: KnowledgeGraph) -> None:
        files = _make_files("src/auth/validate.py", "src/auth/crypto.py")
        process_structure(files, graph)

        contains_rels = graph.get_relationships_by_type(RelType.CONTAINS)

        # Expected: src -> src/auth, src/auth -> validate.py, src/auth -> crypto.py
        assert len(contains_rels) == 3

        # Verify folder -> file relationships exist.
        auth_folder_id = generate_id(NodeLabel.FOLDER, "src/auth")
        validate_file_id = generate_id(NodeLabel.FILE, "src/auth/validate.py")
        crypto_file_id = generate_id(NodeLabel.FILE, "src/auth/crypto.py")

        rel_pairs = {(r.source, r.target) for r in contains_rels}
        assert (auth_folder_id, validate_file_id) in rel_pairs
        assert (auth_folder_id, crypto_file_id) in rel_pairs


class TestNestedFolders:
    """test_nested_folders — parent folder contains child folder."""

    def test_nested_folders(self, graph: KnowledgeGraph) -> None:
        files = _make_files("a/b/c/file.py")
        process_structure(files, graph)

        folder_nodes = graph.get_nodes_by_label(NodeLabel.FOLDER)
        folder_paths = {n.file_path for n in folder_nodes}
        assert folder_paths == {"a", "a/b", "a/b/c"}

        # Verify nesting relationships.
        contains_rels = graph.get_relationships_by_type(RelType.CONTAINS)

        a_id = generate_id(NodeLabel.FOLDER, "a")
        ab_id = generate_id(NodeLabel.FOLDER, "a/b")
        abc_id = generate_id(NodeLabel.FOLDER, "a/b/c")
        file_id = generate_id(NodeLabel.FILE, "a/b/c/file.py")

        rel_pairs = {(r.source, r.target) for r in contains_rels}
        assert (a_id, ab_id) in rel_pairs
        assert (ab_id, abc_id) in rel_pairs
        assert (abc_id, file_id) in rel_pairs
        assert len(contains_rels) == 3


class TestRootLevelFiles:
    """test_root_level_files — files at root level work correctly."""

    def test_root_level_files(self, graph: KnowledgeGraph) -> None:
        files = _make_files("README.md", "setup.py")
        process_structure(files, graph)

        file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)
        assert len(file_nodes) == 2

        # No folders should be created for root-level files.
        folder_nodes = graph.get_nodes_by_label(NodeLabel.FOLDER)
        assert len(folder_nodes) == 0

        # No CONTAINS relationships since there are no parent folders.
        contains_rels = graph.get_relationships_by_type(RelType.CONTAINS)
        assert len(contains_rels) == 0

    def test_root_level_mixed_with_nested(self, graph: KnowledgeGraph) -> None:
        """Root-level files coexist with nested files without errors."""
        files = _make_files("README.md", "src/main.py")
        process_structure(files, graph)

        file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)
        assert len(file_nodes) == 2

        folder_nodes = graph.get_nodes_by_label(NodeLabel.FOLDER)
        assert len(folder_nodes) == 1
        assert folder_nodes[0].file_path == "src"

        # Only one CONTAINS: src -> src/main.py
        contains_rels = graph.get_relationships_by_type(RelType.CONTAINS)
        assert len(contains_rels) == 1


class TestNoDuplicateFolders:
    """test_no_duplicate_folders — same folder from multiple files is created once."""

    def test_no_duplicate_folders(self, graph: KnowledgeGraph) -> None:
        files = _make_files(
            "src/auth/validate.py",
            "src/auth/crypto.py",
            "src/auth/tokens.py",
        )
        process_structure(files, graph)

        folder_nodes = graph.get_nodes_by_label(NodeLabel.FOLDER)
        folder_paths = [n.file_path for n in folder_nodes]

        # "src/auth" must appear exactly once despite three files living there.
        assert folder_paths.count("src/auth") == 1
        assert folder_paths.count("src") == 1


class TestFileNodeProperties:
    """test_file_node_properties — name, content, language set correctly."""

    def test_file_node_properties(self, graph: KnowledgeGraph) -> None:
        files = [
            FileEntry(
                path="src/auth/validate.py",
                content="def validate(): pass",
                language="python",
            )
        ]
        process_structure(files, graph)

        file_id = generate_id(NodeLabel.FILE, "src/auth/validate.py")
        node = graph.get_node(file_id)

        assert node is not None
        assert node.name == "validate.py"
        assert node.file_path == "src/auth/validate.py"
        assert node.content == "def validate(): pass"
        assert node.language == "python"
        assert node.label == NodeLabel.FILE


class TestEmptyFileList:
    """test_empty_file_list — empty input produces empty graph."""

    def test_empty_file_list(self, graph: KnowledgeGraph) -> None:
        process_structure([], graph)

        assert list(graph.iter_nodes()) == []
        assert list(graph.iter_relationships()) == []
        assert graph.stats() == {"nodes": 0, "relationships": 0}
