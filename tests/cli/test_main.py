from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit
from typer.testing import CliRunner

import axon.mcp.server as _mcp_server
from axon import __version__
from axon.cli.main import (
    _initialize_writable_storage,
    _register_in_global_registry,
    _start_host_background,
    app,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def suppress_update_notice(request):
    if request.node.get_closest_marker("allow_update_notice"):
        yield
        return
    with patch("axon.cli.main._maybe_notify_update"):
        yield


class TestVersion:
    def test_version_long_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0


class TestUpdateNotifier:
    @pytest.mark.allow_update_notice
    def test_shows_update_notice_for_normal_command(self) -> None:
        with patch("axon.cli.main._get_latest_version", return_value="9.9.9"):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Update available" in result.output

    @pytest.mark.allow_update_notice
    def test_skips_update_notice_for_serve(self) -> None:
        with patch("axon.cli.main._get_latest_version", return_value="9.9.9"):
            with patch("asyncio.run"):
                result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert "Update available" not in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert f"Axon v{__version__}" in result.output

    def test_version_exit_code(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0


class TestHelp:
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
            "host",
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in --help output"


class TestStatus:
    def test_status_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_status_with_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
    def test_list_calls_handle_list_repos(self) -> None:
        with patch(
            "axon.mcp.tools.handle_list_repos",
            return_value="Indexed repositories (1):\n\n  1. my-project",
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "my-project" in result.output

    def test_list_no_repos(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        with patch(
            "axon.mcp.tools.handle_list_repos",
            return_value="No indexed repositories found.",
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No indexed repositories found" in result.output


class TestClean:
    def test_clean_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_clean_with_force(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert not axon_dir.exists()

    def test_clean_aborted(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(app, ["clean"], input="n\n")
        assert result.exit_code == 0
        assert axon_dir.exists()  # Not deleted


class TestQuery:
    def test_query_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["query", "find classes"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_query_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
    def test_context_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["context", "MyClass"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_context_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
    def test_impact_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["impact", "MyClass.method"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_impact_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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

    def test_impact_default_depth(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
    def test_dead_code_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["dead-code"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_dead_code_with_storage(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
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
    def test_cypher_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["cypher", "MATCH (n) RETURN n"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_cypher_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
    def test_setup_no_flags_shows_both(self) -> None:
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" in result.output
        assert '"axon"' in result.output
        assert '"serve"' in result.output

    def test_setup_claude_only(self) -> None:
        result = runner.invoke(app, ["setup", "--claude"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" not in result.output

    def test_setup_cursor_only(self) -> None:
        result = runner.invoke(app, ["setup", "--cursor"])
        assert result.exit_code == 0
        assert "Cursor" in result.output
        assert "Claude Code" not in result.output

    def test_setup_both_flags(self) -> None:
        result = runner.invoke(app, ["setup", "--claude", "--cursor"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" in result.output


class TestMcp:
    def test_mcp_command_exists(self) -> None:
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP server" in result.output or "stdio" in result.output.lower()

    def test_mcp_calls_server_main(self) -> None:
        import asyncio as real_asyncio

        with patch.object(real_asyncio, "run") as mock_run:
            result = runner.invoke(app, ["mcp"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_accepts_path_argument(self) -> None:
        """PATH argument appears in mcp --help output."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert "PATH" in result.output

    def test_rejects_nonexistent_path(self, tmp_path: Path) -> None:
        """Non-existent path argument causes exit code 1."""
        bogus = tmp_path / "no-such-dir"
        result = runner.invoke(app, ["mcp", str(bogus)])
        assert result.exit_code == 1
        assert "not a directory" in result.output


class TestServe:
    def test_serve_command_exists(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "watch" in result.output.lower()

    def test_serve_without_watch_delegates_to_mcp(self) -> None:
        import asyncio as real_asyncio

        with patch.object(real_asyncio, "run") as mock_run:
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_serve_with_watch_proxies_to_host(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        import asyncio as real_asyncio

        monkeypatch.chdir(tmp_path)
        with patch(
            "axon.cli.main._ensure_host_running",
            return_value={
                "host_url": "http://127.0.0.1:8420",
                "mcp_url": "http://127.0.0.1:8420/mcp",
            },
        ) as mock_ensure:
            with patch.object(real_asyncio, "run") as mock_run:
                result = runner.invoke(app, ["serve", "--watch"])
        assert result.exit_code == 0
        mock_ensure.assert_called_once()
        mock_run.assert_called_once()

    def test_accepts_path_argument(self) -> None:
        """PATH argument appears in serve --help output."""
        result = runner.invoke(app, ["serve", "--help"])
        assert "PATH" in result.output

    def test_rejects_nonexistent_path(self, tmp_path: Path) -> None:
        """Non-existent path argument causes exit code 1."""
        bogus = tmp_path / "no-such-dir"
        result = runner.invoke(app, ["serve", str(bogus)])
        assert result.exit_code == 1
        assert "not a directory" in result.output


class TestHost:
    def test_host_command_exists(self) -> None:
        result = runner.invoke(app, ["host", "--help"])
        assert result.exit_code == 0
        assert "HTTP MCP" in result.output or "shared" in result.output.lower()

    def test_accepts_path_argument(self) -> None:
        """PATH argument appears in host --help output."""
        result = runner.invoke(app, ["host", "--help"])
        assert "PATH" in result.output

    def test_rejects_nonexistent_path(self, tmp_path: Path) -> None:
        """Non-existent path argument causes exit code 1."""
        bogus = tmp_path / "no-such-dir"
        result = runner.invoke(app, ["host", str(bogus)])
        assert result.exit_code == 1
        assert "not a directory" in result.output


class TestUi:
    def test_ui_attaches_to_running_host(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch(
            "axon.cli.main._get_live_host_info",
            return_value={
                "host_url": "http://127.0.0.1:8420",
                "mcp_url": "http://127.0.0.1:8420/mcp",
            },
        ):
            with patch("webbrowser.open") as mock_open:
                result = runner.invoke(app, ["ui"])
        assert result.exit_code == 0
        assert "http://127.0.0.1:8420" in result.output
        mock_open.assert_called_once_with("http://127.0.0.1:8420")

    def test_ui_direct_skips_host_attach(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon.cli.main._get_live_host_info") as mock_host_info:
            with patch(
                "axon.cli.main._initialize_writable_storage",
                return_value=(mock_storage, tmp_path / ".axon", tmp_path / ".axon" / "kuzu"),
            ):
                with patch("axon.web.app.create_app") as mock_create_app:
                    with patch("uvicorn.run") as mock_run:
                        result = runner.invoke(app, ["ui", "--direct", "--no-open"])
        assert result.exit_code == 0
        mock_host_info.assert_not_called()
        mock_create_app.assert_called_once()
        mock_run.assert_called_once()


class TestWritableStorageInitialization:
    def test_requires_database_when_auto_index_disabled(self, tmp_path: Path) -> None:
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        with pytest.raises(Exit):
            _initialize_writable_storage(tmp_path, auto_index=False)

    def test_runs_embedding_migration_for_existing_index(self, tmp_path: Path) -> None:
        axon_dir = tmp_path / ".axon"
        db_path = axon_dir / "kuzu"
        db_path.mkdir(parents=True)
        (db_path / "data.kz").write_text("", encoding="utf-8")
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        mock_storage = MagicMock()
        with patch("axon.cli.main.KuzuBackend", return_value=mock_storage):
            with patch("axon.cli.main.ensure_current_embeddings") as mock_migrate:
                storage, returned_axon_dir, returned_db_path = _initialize_writable_storage(
                    tmp_path,
                    auto_index=False,
                )

        assert storage is mock_storage
        assert returned_axon_dir == axon_dir
        assert returned_db_path == db_path
        mock_storage.initialize.assert_called_once_with(db_path)
        mock_migrate.assert_called_once_with(mock_storage, tmp_path)


class TestWatch:
    def test_watch_command_exists(self) -> None:
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch mode" in result.output or "re-index" in result.output.lower()

    def test_diff_command_exists(self) -> None:
        result = runner.invoke(app, ["diff", "--help"])
        assert result.exit_code == 0
        assert "branch" in result.output.lower()

    def test_accepts_path_argument(self) -> None:
        """PATH argument appears in watch --help output."""
        result = runner.invoke(app, ["watch", "--help"])
        assert "PATH" in result.output

    def test_rejects_nonexistent_path(self, tmp_path: Path) -> None:
        """Non-existent path argument causes exit code 1."""
        bogus = tmp_path / "no-such-dir"
        result = runner.invoke(app, ["watch", str(bogus)])
        assert result.exit_code == 1
        assert "not a directory" in result.output


# Multi-repo registry


class TestRegisterInGlobalRegistry:
    def test_first_registration(self, tmp_path: Path) -> None:
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
        repo_path = tmp_path / "myapp"
        repo_path.mkdir()
        meta = {"name": "myapp", "path": str(repo_path), "stats": {}}

        # Ensure no .axon dir exists
        assert not (tmp_path / ".axon").exists()

        with patch("axon.cli.main.Path.home", return_value=tmp_path):
            _register_in_global_registry(meta, repo_path)

        assert (tmp_path / ".axon" / "repos" / "myapp" / "meta.json").exists()


class TestSetDbPath:
    def test_updates_global(self, monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
        """set_db_path() updates the module-level _db_path global."""
        monkeypatch.setattr(_mcp_server, "_db_path", None)
        custom = tmp_path / ".axon" / "kuzu"
        _mcp_server.set_db_path(custom)
        assert _mcp_server._db_path == custom


class TestStartHostBackground:
    def test_passes_repo_path_in_command(self, tmp_path: Path) -> None:
        """_start_host_background() appends repo_path as the final CLI argument."""
        with patch("subprocess.Popen") as mock_popen, \
                patch("builtins.open", MagicMock()):
            _start_host_background(tmp_path, port=8420)
        cmd = mock_popen.call_args[0][0]
        assert cmd[-1] == str(tmp_path)
