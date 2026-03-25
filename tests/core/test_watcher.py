from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import watchfiles

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from axon.core.ingestion.pipeline import reindex_files, run_pipeline
from axon.core.ingestion.walker import FileEntry, read_file
from axon.core.ingestion.watcher import (
    _compute_dirty_node_ids,
    _get_head_sha,
    _make_watch_filter,
    _reindex_files,
    _run_incremental_global_phases,
)
from axon.core.storage.kuzu_backend import KuzuBackend


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small Python repository for watcher tests."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "app.py").write_text(
        "def hello():\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )

    (src / "utils.py").write_text(
        "def helper():\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def storage(tmp_path: Path) -> KuzuBackend:
    """Provide an initialised KuzuBackend for testing."""
    db_path = tmp_path / "test_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


class TestReadFileEntry:
    def test_reads_python_file(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "src" / "app.py")

        assert entry is not None
        assert entry.path == "src/app.py"
        assert entry.language == "python"
        assert "hello" in entry.content

    def test_returns_none_for_unsupported(self, tmp_repo: Path) -> None:
        readme = tmp_repo / "README.md"
        readme.write_text("# readme", encoding="utf-8")

        entry = read_file(tmp_repo, readme)

        assert entry is None

    def test_returns_none_for_missing(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "nonexistent.py")

        assert entry is None

    def test_returns_none_for_empty(self, tmp_repo: Path) -> None:
        empty = tmp_repo / "empty.py"
        empty.write_text("", encoding="utf-8")

        entry = read_file(tmp_repo, empty)

        assert entry is None


class TestReindexFiles:
    def test_reindex_updates_content(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)

        # Verify initial node exists.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "hello" in node.content

        # Modify the file.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n"
            "    return 'goodbye'\n",
            encoding="utf-8",
        )

        # Re-read and reindex.
        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        # Verify updated node.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "goodbye" in node.content

    def test_reindex_handles_new_symbols(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)

        # Add a new function to the file.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n"
            "    return 'hello'\n"
            "\n"
            "def world():\n"
            "    return 'world'\n",
            encoding="utf-8",
        )

        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        # Both symbols should exist.
        assert storage.get_node("function:src/app.py:hello") is not None
        assert storage.get_node("function:src/app.py:world") is not None

    def test_reindex_removes_deleted_symbols(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)
        assert storage.get_node("function:src/app.py:hello") is not None

        # Remove the function.
        (tmp_repo / "src" / "app.py").write_text(
            "# empty file\nX = 1\n",
            encoding="utf-8",
        )

        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        # Old symbol should be gone.
        assert storage.get_node("function:src/app.py:hello") is None

    def test_reindex_rebuilds_same_file_call_edges_for_new_symbols(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage, embeddings=False)

        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n"
            "    return world()\n"
            "\n"
            "def world():\n"
            "    return 'world'\n",
            encoding="utf-8",
        )

        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        graph = storage.load_graph()
        calls = graph.get_relationships_by_type(RelType.CALLS)
        assert any(
            rel.source == "function:src/app.py:hello"
            and rel.target == "function:src/app.py:world"
            for rel in calls
        )

    def test_reindex_rebuilds_import_edges_for_new_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage, embeddings=False)

        (tmp_repo / "src" / "app.py").write_text(
            "from .helper_new import helper_new\n"
            "\n"
            "def hello():\n"
            "    return helper_new()\n",
            encoding="utf-8",
        )
        (tmp_repo / "src" / "helper_new.py").write_text(
            "def helper_new():\n"
            "    return 42\n",
            encoding="utf-8",
        )

        entries = [
            FileEntry(
                path="src/app.py",
                content=(tmp_repo / "src" / "app.py").read_text(),
                language="python",
            ),
            FileEntry(
                path="src/helper_new.py",
                content=(tmp_repo / "src" / "helper_new.py").read_text(),
                language="python",
            ),
        ]
        reindex_files(entries, tmp_repo, storage)

        graph = storage.load_graph()
        imports = graph.get_relationships_by_type(RelType.IMPORTS)
        assert any(
            rel.source == "file:src/app.py:"
            and rel.target == "file:src/helper_new.py:"
            for rel in imports
        )


class TestWatcherReindexFiles:
    def test_reindexes_changed_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Modify a file.
        app_path = tmp_repo / "src" / "app.py"
        app_path.write_text(
            "def hello():\n    return 'updated'\n",
            encoding="utf-8",
        )

        count, paths = _reindex_files([app_path], tmp_repo, storage)

        assert count == 1
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "updated" in node.content

    def test_skips_ignored_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Create a file in an ignored directory.
        cache_dir = tmp_repo / "__pycache__"
        cache_dir.mkdir()
        cached = cache_dir / "module.cpython-311.pyc"
        cached.write_bytes(b"\x00")

        count, _paths = _reindex_files([cached], tmp_repo, storage)

        assert count == 0

    def test_skips_unsupported_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        readme = tmp_repo / "README.md"
        readme.write_text("# hello", encoding="utf-8")

        count, _paths = _reindex_files([readme], tmp_repo, storage)

        assert count == 0

    def test_handles_deleted_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # File exists in storage but is now deleted from disk.
        deleted_path = tmp_repo / "src" / "app.py"
        assert storage.get_node("file:src/app.py:") is not None

        deleted_path.unlink()

        count, _paths = _reindex_files([deleted_path], tmp_repo, storage)

        # Returns 1: the deleted file was processed (nodes removed from storage).
        assert count == 1

    def test_skips_deleted_files_in_ignored_dirs(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # A file that never existed in the index (lives in an ignored directory).
        ignored_deleted = tmp_repo / "build" / "output.py"

        count, _paths = _reindex_files([ignored_deleted], tmp_repo, storage)

        # The ignored path should not be processed at all.
        assert count == 0

    def test_handles_multiple_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Modify both files.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n    return 'v2'\n",
            encoding="utf-8",
        )
        (tmp_repo / "src" / "utils.py").write_text(
            "def helper():\n    return 42\n",
            encoding="utf-8",
        )

        count, _paths = _reindex_files(
            [tmp_repo / "src" / "app.py", tmp_repo / "src" / "utils.py"],
            tmp_repo,
            storage,
        )

        assert count == 2


class TestWatchFilter:
    """Tests for _make_watch_filter — the gitignore-aware watchfiles filter."""

    def test_allows_regular_source_file(self, tmp_repo: Path) -> None:
        watch_filter = _make_watch_filter(tmp_repo, None)
        src_file = str(tmp_repo / "src" / "app.py")
        assert watch_filter(watchfiles.Change.modified, src_file) is True

    def test_blocks_default_ignored_directory(self, tmp_repo: Path) -> None:
        watch_filter = _make_watch_filter(tmp_repo, None)
        cached = str(tmp_repo / "__pycache__" / "module.cpython-311.pyc")
        assert watch_filter(watchfiles.Change.added, cached) is False

    def test_blocks_gitignore_pattern(self, tmp_repo: Path) -> None:
        # Write a .gitignore that excludes "build/" directory.
        (tmp_repo / ".gitignore").write_text("build/\n", encoding="utf-8")
        gitignore_patterns = ["build/"]
        watch_filter = _make_watch_filter(tmp_repo, gitignore_patterns)
        build_file = str(tmp_repo / "build" / "output.py")
        assert watch_filter(watchfiles.Change.added, build_file) is False

    def test_allows_non_ignored_file_with_gitignore(self, tmp_repo: Path) -> None:
        gitignore_patterns = ["build/", "dist/", "*.log"]
        watch_filter = _make_watch_filter(tmp_repo, gitignore_patterns)
        normal_file = str(tmp_repo / "src" / "main.py")
        assert watch_filter(watchfiles.Change.modified, normal_file) is True

    def test_blocks_node_modules(self, tmp_repo: Path) -> None:
        watch_filter = _make_watch_filter(tmp_repo, None)
        pkg_file = str(tmp_repo / "node_modules" / "lodash" / "index.js")
        assert watch_filter(watchfiles.Change.modified, pkg_file) is False

    def test_handles_path_outside_repo(self, tmp_repo: Path, tmp_path: Path) -> None:
        """Paths outside repo_path are passed through without filtering."""
        watch_filter = _make_watch_filter(tmp_repo, None)
        outside = str(tmp_path / "other" / "file.py")
        # Outside repo_path: our filter defers to DefaultFilter (not hidden/ignored) → True.
        assert watch_filter(watchfiles.Change.modified, outside) is True


class TestGetHeadSha:
    def test_returns_sha_in_git_repo(self, tmp_repo: Path) -> None:
        subprocess.run(["git", "init"], cwd=tmp_repo, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=tmp_repo, capture_output=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_repo,
            capture_output=True,
            env=env,
        )
        sha = _get_head_sha(tmp_repo)
        assert sha is not None
        assert len(sha) == 40

    def test_returns_none_outside_git_repo(self, tmp_path: Path) -> None:
        sha = _get_head_sha(tmp_path)
        assert sha is None


class TestReindexFilesReturnType:
    def test_returns_count_and_paths(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage, embeddings=False)

        changed = [tmp_repo / "src" / "app.py"]
        count, paths = _reindex_files(changed, tmp_repo, storage)
        assert count == 1
        assert "src/app.py" in paths


class TestComputeDirtyNodeIds:
    def test_includes_dirty_file_nodes(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage, embeddings=False)

        graph = storage.load_graph()
        dirty_ids = _compute_dirty_node_ids(graph, {"src/app.py"})
        assert any("app.py" in nid for nid in dirty_ids)

    def test_returns_empty_for_empty_input(self, storage: KuzuBackend) -> None:
        graph = KnowledgeGraph()
        result = _compute_dirty_node_ids(graph, set())
        assert result == set()


class TestRunIncrementalGlobalPhases:
    def test_small_changes_preserve_existing_synthetic_nodes(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage, embeddings=False)

        storage.add_nodes([
            GraphNode(
                id="community:synthetic:test",
                label=NodeLabel.COMMUNITY,
                name="Synthetic Community",
            ),
        ])
        storage.add_relationships([
            GraphRelationship(
                id="member_of:function:src/app.py:hello->community:synthetic:test",
                type=RelType.MEMBER_OF,
                source="function:src/app.py:hello",
                target="community:synthetic:test",
            ),
        ])

        _run_incremental_global_phases(
            storage, tmp_repo, dirty_files={"src/app.py"}, run_coupling=False,
        )

        graph = storage.load_graph()
        assert graph.get_node("community:synthetic:test") is not None
        assert any(
            rel.type == RelType.MEMBER_OF
            and rel.target == "community:synthetic:test"
            for rel in graph.iter_relationships()
        )

    def test_no_stale_synthetic_nodes_after_rerun(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage, embeddings=False)

        # First incremental run — should create communities.
        _run_incremental_global_phases(
            storage,
            tmp_repo,
            dirty_files={"src/app.py", "src/utils.py", "src/third.py"},
            run_coupling=False,
        )
        graph1 = storage.load_graph()
        comm_count_1 = len(list(graph1.get_nodes_by_label(NodeLabel.COMMUNITY)))

        # Second incremental run — old communities should be deleted before new ones.
        _run_incremental_global_phases(
            storage,
            tmp_repo,
            dirty_files={"src/app.py", "src/utils.py", "src/third.py"},
            run_coupling=False,
        )
        graph2 = storage.load_graph()
        comm_count_2 = len(list(graph2.get_nodes_by_label(NodeLabel.COMMUNITY)))

        # Community count should be stable, not doubled.
        assert comm_count_2 == comm_count_1
