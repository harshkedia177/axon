"""File system walker for discovering and reading source files in a repository."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from axon.config.ignore import should_ignore
from axon.config.languages import get_language, is_supported

# FS-2: Maximum file size to index (prevents OOM on oversized files)
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

# FS-3: Maximum directory depth when walking the repo tree.
# rglob() has no depth limit; deeply nested trees (e.g. node_modules symlink
# loops or generated artefacts) can cause excessive resource consumption.
MAX_WALK_DEPTH = 50

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
    repo_str = str(repo_path)

    for dirpath, dirnames, filenames in os.walk(repo_path):
        # FS-3: Enforce depth limit — count path separators relative to root.
        # Mutating dirnames in-place stops os.walk() from descending further.
        depth = dirpath[len(repo_str):].count(os.sep)
        if depth >= MAX_WALK_DEPTH:
            dirnames.clear()
            continue

        for filename in filenames:
            file_path = Path(dirpath) / filename

            if not file_path.is_file():
                continue

            # FS-1: Prevent symlink traversal outside repo boundary
            if file_path.is_symlink():
                continue
            try:
                resolved = file_path.resolve()
                if not resolved.is_relative_to(repo_path):
                    continue
            except (OSError, ValueError):
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
    when the file is empty, when an OS-level error occurs, or when the file
    exceeds :data:`MAX_FILE_SIZE`.
    """
    relative = file_path.relative_to(repo_path)

    # FS-2: Skip oversized files to prevent OOM
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return None
    except OSError:
        return None

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
