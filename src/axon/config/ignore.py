"""Ignore-pattern handling for Axon's file discovery."""

from __future__ import annotations

import fnmatch
from pathlib import Path

DEFAULT_IGNORE_PATTERNS: frozenset[str] = frozenset(
    {
        # Directories
        "node_modules",
        "__pycache__",
        ".git",
        ".axon",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "egg-info",
        ".eggs",
        "coverage",
        "htmlcov",
        # Files (exact names)
        ".DS_Store",
        ".coverage",
        "package-lock.json",
        "yarn.lock",
        "uv.lock",
        "poetry.lock",
        # File globs
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dylib",
        "*.min.js",
        "*.bundle.js",
        "*.map",
    }
)

# Separate glob patterns (contain wildcards) from literal names at module load
# so we only compute this once.
_GLOB_PATTERNS: frozenset[str] = frozenset(
    p for p in DEFAULT_IGNORE_PATTERNS if "*" in p or "?" in p
)
_LITERAL_PATTERNS: frozenset[str] = DEFAULT_IGNORE_PATTERNS - _GLOB_PATTERNS


def _matches_default_patterns(path: Path) -> bool:
    """Check whether *path* (relative) matches any default ignore pattern."""
    for part in path.parts:
        if part in _LITERAL_PATTERNS:
            return True
        # Also check globs against every component (e.g. *.pyc as a directory
        # — unlikely but consistent).
        for pattern in _GLOB_PATTERNS:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


_pathspec_cache: dict[tuple[str, ...], object] = {}


def _matches_gitignore(path: Path, gitignore_patterns: list[str]) -> bool:
    """Check *path* against a list of gitignore-style patterns.

    Uses ``pathspec`` when available for full gitignore semantics; falls back to
    fnmatch per-pattern otherwise.  The compiled pathspec is cached by the
    pattern content so it is only built once per unique pattern set.
    """
    if not gitignore_patterns:
        return False

    try:
        import pathspec

        cache_key = tuple(gitignore_patterns)
        spec = _pathspec_cache.get(cache_key)
        if spec is None:
            spec = pathspec.PathSpec.from_lines("gitignore", gitignore_patterns)
            _pathspec_cache[cache_key] = spec
        return spec.match_file(str(path))  # type: ignore[union-attr]
    except ImportError:  # pragma: no cover — pathspec is a declared dependency
        path_str = str(path)
        for pattern in gitignore_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(path.name, pattern):
                return True
        return False


def should_ignore(
    path: str | Path,
    gitignore_patterns: list[str] | None = None,
) -> bool:
    """Return ``True`` if *path* should be ignored during file discovery.

    Parameters
    ----------
    path:
        A relative file path (e.g. ``src/main.py`` or ``node_modules/pkg/index.js``).
    gitignore_patterns:
        Optional list of gitignore-style patterns loaded via :func:`load_gitignore`.
    """
    p = Path(path)

    if _matches_default_patterns(p):
        return True

    if gitignore_patterns and _matches_gitignore(p, gitignore_patterns):
        return True

    return False


def load_gitignore(repo_path: Path) -> list[str]:
    """Read ``.gitignore`` from *repo_path* and return a list of patterns.

    Blank lines and comments (lines starting with ``#``) are stripped.
    Returns an empty list when the file does not exist.
    """
    gitignore = repo_path / ".gitignore"
    if not gitignore.is_file():
        return []

    lines: list[str] = []
    text = gitignore.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines
