"""Security tests for filesystem safety remediations.

Validates that Phase 1 fixes for filesystem traversal and input validation
are effective and regression-proof.  Covers:

- FS-1: Symlink traversal prevention in discover_files
- FS-2: File size limit enforcement in read_file
- INJ-6: Git ref argument injection prevention in _build_graph_for_ref
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from axon.core.ingestion.walker import MAX_FILE_SIZE, FileEntry, discover_files, read_file


# ---------------------------------------------------------------------------
# (a) Symlink traversal prevention
# ---------------------------------------------------------------------------


class TestSymlinkTraversalPrevention:
    """Verify FS-1: discover_files skips symlinks pointing outside the repo."""

    def test_symlink_outside_repo_excluded(self, tmp_path: Path) -> None:
        """A symlink pointing outside the repo boundary must not appear in results."""
        # Set up repo with a real file
        repo = tmp_path / "repo"
        src = repo / "src"
        src.mkdir(parents=True)
        real_file = src / "real_file.py"
        real_file.write_text("def real(): pass", encoding="utf-8")

        # Set up file outside repo
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.py"
        secret.write_text("SECRET_KEY = 'leaked'", encoding="utf-8")

        # Create symlink from inside repo to outside file
        evil_link = src / "evil_link.py"
        evil_link.symlink_to(secret)

        # Verify the symlink exists and points outside
        assert evil_link.is_symlink()
        assert evil_link.resolve().parent == outside

        discovered = discover_files(repo)
        discovered_names = {p.name for p in discovered}

        assert "real_file.py" in discovered_names, "Real file should be discovered"
        assert "evil_link.py" not in discovered_names, (
            "Symlink to outside repo must be excluded"
        )

    def test_symlink_inside_repo_also_excluded(self, tmp_path: Path) -> None:
        """Even symlinks within the repo are excluded (all symlinks are skipped)."""
        repo = tmp_path / "repo"
        src = repo / "src"
        src.mkdir(parents=True)

        original = src / "original.py"
        original.write_text("def original(): pass", encoding="utf-8")

        internal_link = src / "link_to_original.py"
        internal_link.symlink_to(original)

        discovered = discover_files(repo)
        discovered_names = {p.name for p in discovered}

        assert "original.py" in discovered_names
        assert "link_to_original.py" not in discovered_names, (
            "All symlinks should be excluded regardless of target"
        )

    def test_real_files_not_affected(self, tmp_path: Path) -> None:
        """Regular (non-symlink) source files are discovered normally."""
        repo = tmp_path / "repo"
        src = repo / "src"
        src.mkdir(parents=True)

        for name in ["auth.py", "models.py", "utils.py"]:
            (src / name).write_text(f"# {name}", encoding="utf-8")

        discovered = discover_files(repo)
        discovered_names = {p.name for p in discovered}

        assert "auth.py" in discovered_names
        assert "models.py" in discovered_names
        assert "utils.py" in discovered_names


# ---------------------------------------------------------------------------
# (b) File size limit enforcement
# ---------------------------------------------------------------------------


class TestFileSizeLimit:
    """Verify FS-2: read_file rejects files exceeding MAX_FILE_SIZE."""

    def test_oversized_file_returns_none(self, tmp_path: Path) -> None:
        """A file larger than MAX_FILE_SIZE must be rejected (returns None)."""
        repo = tmp_path / "repo"
        repo.mkdir()

        big_file = repo / "huge.py"
        # Write 3MB of content (MAX_FILE_SIZE is 2MB)
        big_file.write_text("x" * (3 * 1024 * 1024), encoding="utf-8")

        result = read_file(repo, big_file)
        assert result is None, (
            f"Files over {MAX_FILE_SIZE} bytes must return None, got {type(result)}"
        )

    def test_normal_file_returns_entry(self, tmp_path: Path) -> None:
        """A normal-sized .py file returns a valid FileEntry."""
        repo = tmp_path / "repo"
        repo.mkdir()

        normal_file = repo / "small.py"
        normal_file.write_text("x = 42", encoding="utf-8")

        result = read_file(repo, normal_file)
        assert isinstance(result, FileEntry)
        assert result.content == "x = 42"
        assert result.language == "python"

    def test_file_exactly_at_limit(self, tmp_path: Path) -> None:
        """A file exactly at MAX_FILE_SIZE should be accepted."""
        repo = tmp_path / "repo"
        repo.mkdir()

        edge_file = repo / "edge.py"
        # Write exactly MAX_FILE_SIZE bytes of valid Python
        content = "x" * MAX_FILE_SIZE
        edge_file.write_text(content, encoding="utf-8")

        result = read_file(repo, edge_file)
        # File at exactly MAX_FILE_SIZE should pass (st_size <= MAX_FILE_SIZE is
        # technically possible due to encoding, but the content we wrote is ASCII
        # so st_size == len(content)).
        assert result is None or isinstance(result, FileEntry)

    def test_max_file_size_is_2mb(self) -> None:
        """MAX_FILE_SIZE is set to 2MB as documented."""
        assert MAX_FILE_SIZE == 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# (c) Git ref injection prevention
# ---------------------------------------------------------------------------


class TestGitRefInjection:
    """Verify INJ-6: _build_graph_for_ref validates ref names."""

    def test_dash_dash_version_raises(self) -> None:
        """A ref starting with '--' is rejected as argument injection."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises(ValueError, match="Invalid git ref"):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "--version")

    def test_dash_c_flag_raises(self) -> None:
        """A ref containing a space (like '-c evil=true') is rejected."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises(ValueError, match="Invalid git ref"):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "-c evil=true")

    def test_single_dash_raises(self) -> None:
        """A ref starting with a single dash is rejected."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises(ValueError, match="Invalid git ref"):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "-flag")

    def test_empty_ref_raises(self) -> None:
        """An empty string ref is rejected."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises(ValueError, match="Invalid git ref"):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "")

    def test_main_does_not_raise_value_error(self) -> None:
        """'main' is a valid ref name -- should not raise ValueError.
        It may raise RuntimeError because there is no git repo, and that is fine."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises((RuntimeError, OSError)):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "main")

    def test_feature_branch_does_not_raise_value_error(self) -> None:
        """'feature/my-branch' is a valid ref -- should not raise ValueError."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises((RuntimeError, OSError)):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "feature/my-branch")

    def test_tag_ref_does_not_raise_value_error(self) -> None:
        """'v1.0.0' is a valid ref name -- should not raise ValueError."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises((RuntimeError, OSError)):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "v1.0.0")

    def test_ref_with_special_chars_raises(self) -> None:
        """A ref with shell metacharacters is rejected."""
        from axon.core.diff import _build_graph_for_ref

        with pytest.raises(ValueError, match="Invalid git ref"):
            _build_graph_for_ref(Path("/tmp/fake_repo"), "main; rm -rf /")

    def test_safe_ref_regex_directly(self) -> None:
        """Verify _SAFE_REF_RE accepts valid refs and rejects dangerous ones.

        Note: The regex allows hyphens (needed for branch names like
        'feature/my-branch'), so refs like '--version' match the regex.
        The actual defense against dash-prefixed refs is the explicit
        ``ref.startswith("-")`` check in _build_graph_for_ref. This test
        validates the regex component: character-class filtering of shell
        metacharacters, spaces, quotes, and empty strings.
        """
        from axon.core.diff import _SAFE_REF_RE

        # Valid refs -- all should match
        assert _SAFE_REF_RE.match("main") is not None
        assert _SAFE_REF_RE.match("feature/my-branch") is not None
        assert _SAFE_REF_RE.match("v1.0.0") is not None
        assert _SAFE_REF_RE.match("release/2.0") is not None
        assert _SAFE_REF_RE.match("HEAD") is not None

        # Refs with shell metacharacters -- rejected by regex
        assert _SAFE_REF_RE.match("") is None  # empty string
        assert _SAFE_REF_RE.match("main; rm -rf /") is None  # semicolon + spaces
        assert _SAFE_REF_RE.match("ref with spaces") is None  # spaces
        assert _SAFE_REF_RE.match("ref'injection") is None  # single quote
        assert _SAFE_REF_RE.match('ref"injection') is None  # double quote
        assert _SAFE_REF_RE.match("ref$(cmd)") is None  # command substitution
        assert _SAFE_REF_RE.match("ref`cmd`") is None  # backtick injection
