from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from axon.core.embeddings.embedder import _DEFAULT_MODEL
from axon.core.ingestion.watcher import ensure_current_embeddings


def test_needs_reembed_model_mismatch() -> None:
    meta = {"embedding_model": "BAAI/bge-small-en-v1.5"}
    assert meta.get("embedding_model") != _DEFAULT_MODEL


def test_needs_reembed_missing_key() -> None:
    meta = {"version": "1.0.0", "stats": {}}
    assert meta.get("embedding_model") is None


def test_no_reembed_when_matching() -> None:
    meta = {"embedding_model": _DEFAULT_MODEL}
    assert meta.get("embedding_model") == _DEFAULT_MODEL


def test_ensure_current_embeddings_reembeds_and_updates_meta(tmp_path) -> None:
    repo_path = tmp_path
    axon_dir = repo_path / ".axon"
    axon_dir.mkdir()
    meta_path = axon_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"embedding_model": "BAAI/bge-small-en-v1.5"}) + "\n",
        encoding="utf-8",
    )

    storage = MagicMock()
    storage.load_graph.return_value = object()

    with patch("axon.core.ingestion.watcher.embed_graph", return_value={"node-1": [0.1, 0.2]}):
        migrated = ensure_current_embeddings(storage, repo_path)

    assert migrated is True
    storage.load_graph.assert_called_once_with()
    storage.store_embeddings.assert_called_once_with({"node-1": [0.1, 0.2]})
    updated_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert updated_meta["embedding_model"] == _DEFAULT_MODEL


def test_ensure_current_embeddings_noop_when_model_matches(tmp_path) -> None:
    repo_path = tmp_path
    axon_dir = repo_path / ".axon"
    axon_dir.mkdir()
    (axon_dir / "meta.json").write_text(
        json.dumps({"embedding_model": _DEFAULT_MODEL}) + "\n",
        encoding="utf-8",
    )

    storage = MagicMock()

    migrated = ensure_current_embeddings(storage, repo_path)

    assert migrated is False
    storage.load_graph.assert_not_called()
    storage.store_embeddings.assert_not_called()
