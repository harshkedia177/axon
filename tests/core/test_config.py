"""Tests for axon.config.ignore and axon.config.languages."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.config.ignore import (
    DEFAULT_IGNORE_PATTERNS,
    load_gitignore,
    should_ignore,
)
from axon.config.languages import (
    SUPPORTED_EXTENSIONS,
    get_language,
    is_supported,
)


# ---------------------------------------------------------------------------
# ignore.py tests
# ---------------------------------------------------------------------------


class TestShouldIgnore:
    """Tests for should_ignore()."""

    def test_node_modules_subpath(self) -> None:
        assert should_ignore("node_modules/foo.py") is True

    def test_pycache_subpath(self) -> None:
        assert should_ignore("src/__pycache__/foo.pyc") is True

    def test_pyc_glob_at_root(self) -> None:
        assert should_ignore("foo.pyc") is True

    def test_normal_file_not_ignored(self) -> None:
        assert should_ignore("src/main.py") is False

    def test_git_directory(self) -> None:
        assert should_ignore(".git/config") is True

    def test_ds_store(self) -> None:
        assert should_ignore(".DS_Store") is True

    def test_so_extension(self) -> None:
        assert should_ignore("lib/native.so") is True

    def test_min_js(self) -> None:
        assert should_ignore("static/app.min.js") is True

    def test_bundle_js(self) -> None:
        assert should_ignore("dist/bundle.bundle.js") is True

    def test_lock_files(self) -> None:
        assert should_ignore("package-lock.json") is True
        assert should_ignore("yarn.lock") is True
        assert should_ignore("uv.lock") is True
        assert should_ignore("poetry.lock") is True

    def test_deeply_nested_ignored_dir(self) -> None:
        assert should_ignore("a/b/c/node_modules/d/e.js") is True

    def test_venv_directory(self) -> None:
        assert should_ignore(".venv/lib/python3.12/site.py") is True
        assert should_ignore("venv/bin/activate") is True

    def test_gitignore_patterns(self) -> None:
        patterns = ["*.log", "tmp/"]
        assert should_ignore("debug.log", gitignore_patterns=patterns) is True
        assert should_ignore("tmp/cache", gitignore_patterns=patterns) is True
        assert should_ignore("src/main.py", gitignore_patterns=patterns) is False

    def test_gitignore_none_patterns(self) -> None:
        assert should_ignore("src/main.py", gitignore_patterns=None) is False

    def test_gitignore_empty_patterns(self) -> None:
        assert should_ignore("src/main.py", gitignore_patterns=[]) is False


class TestLoadGitignore:
    """Tests for load_gitignore()."""

    def test_reads_gitignore(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(
            "# comment\n"
            "*.log\n"
            "\n"
            "  tmp/  \n"
            "dist/\n",
            encoding="utf-8",
        )
        patterns = load_gitignore(tmp_path)
        assert patterns == ["*.log", "tmp/", "dist/"]

    def test_no_gitignore(self, tmp_path: Path) -> None:
        patterns = load_gitignore(tmp_path)
        assert patterns == []

    def test_empty_gitignore(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("", encoding="utf-8")
        patterns = load_gitignore(tmp_path)
        assert patterns == []

    def test_comments_only(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("# just a comment\n# another\n", encoding="utf-8")
        patterns = load_gitignore(tmp_path)
        assert patterns == []


class TestDefaultIgnorePatterns:
    """Sanity checks on the constant itself."""

    def test_is_frozenset(self) -> None:
        assert isinstance(DEFAULT_IGNORE_PATTERNS, frozenset)

    def test_contains_expected_entries(self) -> None:
        assert "node_modules" in DEFAULT_IGNORE_PATTERNS
        assert "*.pyc" in DEFAULT_IGNORE_PATTERNS
        assert ".git" in DEFAULT_IGNORE_PATTERNS


# ---------------------------------------------------------------------------
# languages.py tests
# ---------------------------------------------------------------------------


class TestGetLanguage:
    """Tests for get_language()."""

    def test_python(self) -> None:
        assert get_language("src/main.py") == "python"

    def test_typescript_ts(self) -> None:
        assert get_language("components/App.ts") == "typescript"

    def test_typescript_tsx(self) -> None:
        assert get_language("components/App.tsx") == "tsx"

    def test_javascript_js(self) -> None:
        assert get_language("index.js") == "javascript"

    def test_javascript_jsx(self) -> None:
        assert get_language("App.jsx") == "javascript"

    def test_javascript_mjs(self) -> None:
        assert get_language("config.mjs") == "javascript"

    def test_javascript_cjs(self) -> None:
        assert get_language("config.cjs") == "javascript"

    def test_unsupported_md(self) -> None:
        assert get_language("README.md") is None

    def test_unsupported_json(self) -> None:
        assert get_language("package.json") is None

    def test_unsupported_txt(self) -> None:
        assert get_language("notes.txt") is None

    def test_no_extension(self) -> None:
        assert get_language("Makefile") is None

    def test_accepts_path_object(self) -> None:
        assert get_language(Path("src/app.py")) == "python"


class TestIsSupported:
    """Tests for is_supported()."""

    def test_supported_py(self) -> None:
        assert is_supported("main.py") is True

    def test_supported_ts(self) -> None:
        assert is_supported("app.ts") is True

    def test_not_supported_md(self) -> None:
        assert is_supported("README.md") is False

    def test_not_supported_json(self) -> None:
        assert is_supported("data.json") is False

    def test_accepts_path_object(self) -> None:
        assert is_supported(Path("src/module.tsx")) is True


class TestSupportedExtensions:
    """Sanity checks on the constant."""

    def test_is_dict(self) -> None:
        assert isinstance(SUPPORTED_EXTENSIONS, dict)

    def test_all_values_are_strings(self) -> None:
        for ext, lang in SUPPORTED_EXTENSIONS.items():
            assert isinstance(ext, str)
            assert isinstance(lang, str)
            assert ext.startswith(".")
