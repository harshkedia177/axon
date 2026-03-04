"""File system walker for discovering and reading source files in a repository."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from axon.config.ignore import should_ignore
from axon.config.languages import get_language, is_supported


@dataclass
class FileEntry:
    """A source file discovered during walking."""

    path: str  # relative path from repo root (e.g., "src/auth/validate.py")
    content: str  # full file content
    language: str  # "python", "typescript", "javascript"

def discover_files(
    repo_path: Path,
    gitignore_patterns: list[str] | None = None,
) -> list[Path]:
    """Discover supported source file paths without reading their content.

    Walks *repo_path* recursively and returns paths that are not ignored and
    have a supported language extension.  Useful for incremental indexing where
    you want to check paths before reading.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to walk.
    gitignore_patterns:
        Optional list of gitignore-style patterns (e.g. from
        :func:`axon.config.ignore.load_gitignore`).

    Returns
    -------
    list[Path]
        List of absolute :class:`Path` objects for each discovered file.
    """
    repo_path = repo_path.resolve()
    discovered: list[Path] = []

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(repo_path)

        if should_ignore(str(relative), gitignore_patterns):
            continue

        if not is_supported(file_path):
            continue

        discovered.append(file_path)

    return discovered

def read_file(repo_path: Path, file_path: Path) -> FileEntry | None:
    """Read a single file and return a :class:`FileEntry`, or ``None`` on failure.

    Returns ``None`` when the file cannot be decoded as UTF-8 (binary files),
    when the file is empty, or when an OS-level error occurs.
    """
    relative = file_path.relative_to(repo_path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError, OSError):
        return None

    if not content:
        return None

    language = get_language(file_path)
    if language is None:
        return None

    return FileEntry(
        path=str(relative),
        content=content,
        language=language,
    )

def walk_repo(
    repo_path: Path,
    gitignore_patterns: list[str] | None = None,
    max_workers: int = 8,
) -> list[FileEntry]:
    """Walk a repository and return all supported source files with their content.

    Discovers files using the same filtering logic as :func:`discover_files`,
    then reads their content in parallel using a :class:`ThreadPoolExecutor`.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to walk.
    gitignore_patterns:
        Optional list of gitignore-style patterns (e.g. from
        :func:`axon.config.ignore.load_gitignore`).
    max_workers:
        Maximum number of threads for parallel file reading.  Defaults to 8.

    Returns
    -------
    list[FileEntry]
        Sorted (by path) list of :class:`FileEntry` objects for every
        discovered source file.
    """
    repo_path = repo_path.resolve()
    file_paths = discover_files(repo_path, gitignore_patterns)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda fp: read_file(repo_path, fp), file_paths)

    entries = [entry for entry in results if entry is not None]
    entries.sort(key=lambda e: e.path)
    return entries
