"""Language detection based on file extensions."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}


def get_language(file_path: str | Path) -> str | None:
    """Return the language name for *file_path* based on its extension.

    Returns ``None`` when the extension is not in :data:`SUPPORTED_EXTENSIONS`.
    """
    suffix = Path(file_path).suffix
    return SUPPORTED_EXTENSIONS.get(suffix)


def is_supported(file_path: str | Path) -> bool:
    """Return ``True`` if *file_path* has a supported extension."""
    return Path(file_path).suffix in SUPPORTED_EXTENSIONS
