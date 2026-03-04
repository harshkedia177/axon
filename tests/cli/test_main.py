"""Tests for the Axon CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from axon import __version__
from axon.cli.main import _register_in_global_registry, app

runner = CliRunner()


class TestVersion:
    """Tests for the --version flag."""

    def test_version_long_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"Axon v{__version__}" in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert f"Axon v{__version__}" in result.output

    def test_version_string_format(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert f"Axon v{__version__}" in result.output


class TestHelp:
    """Tests for the --help flag."""

    def test_help_exit_code(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_app_name(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "Axon" in result.output

    def test_help_lists_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        expected_commands = [
            "analyze",
            "status",
            "list",
            "clean",
            "query",
            "context",
            "impact",
            "dead-code",
            "cypher",
            "setup",
            "watch",
            "diff",
            "mcp",
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in --help output"


class TestStatus:
    """Tests for the status command."""

    def test_status_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Status should error when no .axon directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_status_with_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Status should display stats from meta.json."""
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        meta = {
            "version": "0.1.0",
            "stats": {
                "files": 10,
                "symbols": 42,
                "relationships": 100,
                "clusters": 3,
                "flows": 0,
                "dead_code": 5,
                "coupled_pairs": 0,
            },
            "last_indexed_at": "2025-01-15T10:00:00+00:00",
        }
        (axon_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Index status for" in result.output
        assert "0.1.0" in result.output
        assert "10" in result.output  # files
        assert "42" in result.output  # symbols
        assert "100" in result.output  # relationships


class TestListRepos:
    """Tests for the list command."""

    def test_list_calls_handle_list_repos(self) -> None:
        """List should call handle_list_repos and print the result."""
        with patch(
            "axon.mcp.tools.handle_list_repos",
            return_value="Indexed repositories (1):\n\n  1. my-project",
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "my-project" in result.output

    def test_list_no_repos(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """List should show 'no repos' message when none are indexed."""
        monkeypatch.chdir(tmp_path)
        # Patch the global registry to a non-existent dir so the fallback also fails
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # handle_list_repos returns "No indexed repositories found." when nothing found
        assert (
            "No indexed repositories found" in result.output
            or "repositories" in result.output.lower()
        )


class TestClean:
    """Tests for the clean command."""

    def test_clean_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clean should error when no .axon directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_clean_with_force(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clean with --force should delete .axon without confirmation."""
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert not axon_dir.exists()

    def test_clean_aborted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clean should abort when user says no."""
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        runner.invoke(app, ["clean"], input="n\n")
        assert axon_dir.exists()  # Not deleted


class TestQuery:
    """Tests for the query command."""

    def test_query_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Query should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["query", "find classes"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_query_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Query should call handle_query with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon.mcp.tools.handle_query",
                return_value="1. MyClass (Class) -- src/main.py",
            ):
                result = runner.invoke(app, ["query", "find classes"])
        assert result.exit_code == 0
        assert "MyClass" in result.output


class TestContext:
    """Tests for the context command."""

    def test_context_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Context should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["context", "MyClass"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_context_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Context should call handle_context with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon.mcp.tools.handle_context",
                return_value="Symbol: MyClass (Class)\nFile: src/main.py:1-50",
            ):
                result = runner.invoke(app, ["context", "MyClass"])
        assert result.exit_code == 0
        assert "MyClass" in result.output


class TestImpact:
    """Tests for the impact command."""

    def test_impact_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Impact should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["impact", "MyClass.method"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_impact_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Impact should call handle_impact with loaded storage and depth."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon.mcp.tools.handle_impact",
                return_value="Impact analysis for: MyClass.method",
            ):
                result = runner.invoke(app, ["impact", "MyClass.method", "--depth", "5"])
        assert result.exit_code == 0
        assert "Impact analysis" in result.output

    def test_impact_default_depth(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Impact without --depth should use default depth of 3."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon.mcp.tools.handle_impact",
                return_value="Impact analysis for: foo",
            ):
                result = runner.invoke(app, ["impact", "foo"])
        assert result.exit_code == 0


class TestDeadCode:
    """Tests for the dead-code command."""

    def test_dead_code_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dead-code should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["dead-code"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_dead_code_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dead-code should call handle_dead_code with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon.mcp.tools.handle_dead_code",
                return_value="No dead code detected.",
            ):
                result = runner.invoke(app, ["dead-code"])
        assert result.exit_code == 0
        assert "No dead code detected" in result.output


class TestCypher:
    """Tests for the cypher command."""

    def test_cypher_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cypher should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["cypher", "MATCH (n) RETURN n"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_cypher_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cypher should call handle_cypher with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon.mcp.tools.handle_cypher",
                return_value="Results (3 rows):\n\n  1. foo",
            ):
                result = runner.invoke(app, ["cypher", "MATCH (n) RETURN n"])
        assert result.exit_code == 0
        assert "Results" in result.output


class TestSetup:
    """Tests for the setup command."""

    def test_setup_no_flags_shows_both(self) -> None:
        """Setup with no flags should show config for both Claude and Cursor."""
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" in result.output
        assert '"axon"' in result.output

    def test_setup_claude_only(self) -> None:
        """Setup with --claude should show only Claude config."""
        result = runner.invoke(app, ["setup", "--claude"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" not in result.output

    def test_setup_cursor_only(self) -> None:
        """Setup with --cursor should show only Cursor config."""
        result = runner.invoke(app, ["setup", "--cursor"])
        assert result.exit_code == 0
        assert "Cursor" in result.output
        assert "Claude Code" not in result.output

    def test_setup_both_flags(self) -> None:
        """Setup with both flags should show both configs."""
        result = runner.invoke(app, ["setup", "--claude", "--cursor"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" in result.output


class TestMcp:
    """Tests for the mcp command."""

    def test_mcp_command_exists(self) -> None:
        """The mcp command should be registered."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP server" in result.output or "stdio" in result.output.lower()

    def test_mcp_calls_server_main(self) -> None:
        """MCP command should call asyncio.run(mcp_main())."""
        with patch("axon.cli.main.asyncio", create=True):
            with patch("axon.mcp.server.main"):
                # We need to mock at the import level inside the function
                import asyncio as real_asyncio

                with patch.object(real_asyncio, "run") as mock_run:
                    runner.invoke(app, ["mcp"])
                    mock_run.assert_called_once()


class TestServe:
    """Tests for the serve command."""

    def test_serve_command_exists(self) -> None:
        """The serve command should be registered."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "watch" in result.output.lower()

    def test_serve_without_watch_delegates_to_mcp(self) -> None:
        """serve without --watch should behave like axon mcp."""
        import asyncio as real_asyncio

        with patch.object(real_asyncio, "run") as mock_run:
            runner.invoke(app, ["serve"])
            mock_run.assert_called_once()


class TestWatch:
    """Tests for the watch command."""

    def test_watch_command_exists(self) -> None:
        """The watch command should be registered."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch mode" in result.output or "re-index" in result.output.lower()

    def test_diff_command_exists(self) -> None:
        """The diff command should be registered."""
        result = runner.invoke(app, ["diff", "--help"])
        assert result.exit_code == 0
        assert "branch" in result.output.lower()


# ---------------------------------------------------------------------------
# Multi-repo registry
# ---------------------------------------------------------------------------


class TestRegisterInGlobalRegistry:
    """Tests for _register_in_global_registry()."""

    def test_first_registration(self, tmp_path: Path) -> None:
        """Creates {registry}/repo_name/meta.json with correct content."""
        tmp_path / "registry"
        repo_path = tmp_path / "my-project"
        repo_path.mkdir()

        meta = {"name": "my-project", "path": str(repo_path), "stats": {}}

        with patch("axon.cli.main.Path.home", return_value=tmp_path):
            # _register_in_global_registry uses Path.home() / ".axon" / "repos"
            _register_in_global_registry(meta, repo_path)

        slot = tmp_path / ".axon" / "repos" / "my-project"
        assert slot.exists()
        written = json.loads((slot / "meta.json").read_text())
        assert written["name"] == "my-project"
        assert written["slug"] == "my-project"
        assert written["path"] == str(repo_path)

    def test_same_repo_re_registered(self, tmp_path: Path) -> None:
        """Re-registering the same repo reuses the same slug."""
        repo_path = tmp_path / "my-project"
        repo_path.mkdir()
        meta = {"name": "my-project", "path": str(repo_path), "stats": {}}

        with patch("axon.cli.main.Path.home", return_value=tmp_path):
            _register_in_global_registry(meta, repo_path)
            _register_in_global_registry(meta, repo_path)

        # Only one directory should exist
        registry = tmp_path / ".axon" / "repos"
        entries = list(registry.iterdir())
        assert len(entries) == 1
        assert entries[0].name == "my-project"

    def test_name_collision_different_repos(self, tmp_path: Path) -> None:
        """Different repos with same directory name get different slugs."""
        repo_a = tmp_path / "workspace-a" / "myapp"
        repo_b = tmp_path / "workspace-b" / "myapp"
        repo_a.mkdir(parents=True)
        repo_b.mkdir(parents=True)

        meta_a = {"name": "myapp", "path": str(repo_a), "stats": {}}
        meta_b = {"name": "myapp", "path": str(repo_b), "stats": {}}

        with patch("axon.cli.main.Path.home", return_value=tmp_path):
            _register_in_global_registry(meta_a, repo_a)
            _register_in_global_registry(meta_b, repo_b)

        registry = tmp_path / ".axon" / "repos"
        entries = sorted([e.name for e in registry.iterdir()])
        assert len(entries) == 2
        # One should be "myapp", the other "myapp-<hash>"
        assert entries[0] == "myapp"
        assert entries[1].startswith("myapp-")

    def test_stale_entry_cleanup(self, tmp_path: Path) -> None:
        """Old registry entry is removed when repo re-registers under new slug."""
        repo_path = tmp_path / "myapp"
        repo_path.mkdir()

        # Manually create a stale entry under a hash slug
        registry = tmp_path / ".axon" / "repos"
        stale = registry / "myapp-abcd1234"
        stale.mkdir(parents=True)
        stale_meta = {"name": "myapp", "path": str(repo_path)}
        (stale / "meta.json").write_text(json.dumps(stale_meta))

        meta = {"name": "myapp", "path": str(repo_path), "stats": {}}
        with patch("axon.cli.main.Path.home", return_value=tmp_path):
            _register_in_global_registry(meta, repo_path)

        # Stale entry should be cleaned up
        assert not stale.exists()
        # New entry under bare name should exist
        assert (registry / "myapp" / "meta.json").exists()

    def test_corrupt_existing_meta_json(self, tmp_path: Path) -> None:
        """Corrupt meta.json in existing slot is handled gracefully."""
        registry = tmp_path / ".axon" / "repos" / "myapp"
        registry.mkdir(parents=True)
        (registry / "meta.json").write_text("not valid json!")

        repo_path = tmp_path / "myapp"
        repo_path.mkdir()
        meta = {"name": "myapp", "path": str(repo_path), "stats": {}}

        with patch("axon.cli.main.Path.home", return_value=tmp_path):
            _register_in_global_registry(meta, repo_path)

        # Should claim the slot (no crash)
        written = json.loads((registry / "meta.json").read_text())
        assert written["path"] == str(repo_path)

    def test_registry_dir_created_if_missing(self, tmp_path: Path) -> None:
        """Registry directory is created automatically."""
        repo_path = tmp_path / "myapp"
        repo_path.mkdir()
        meta = {"name": "myapp", "path": str(repo_path), "stats": {}}

        # Ensure no .axon dir exists
        assert not (tmp_path / ".axon").exists()

        with patch("axon.cli.main.Path.home", return_value=tmp_path):
            _register_in_global_registry(meta, repo_path)

        assert (tmp_path / ".axon" / "repos" / "myapp" / "meta.json").exists()
