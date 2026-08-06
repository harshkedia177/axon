from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from axon.core.embeddings.embedder import _DEFAULT_MODEL, get_effective_embedding_model_name
from axon.core.ingestion.watcher import ensure_current_embeddings
from axon.core.storage.base import get_embedding_dimensions


def test_needs_reembed_model_mismatch() -> None:
    meta = {"embedding_model": "BAAI/bge-small-en-v1.5"}
    assert meta.get("embedding_model") != _DEFAULT_MODEL


def test_needs_reembed_missing_key() -> None:
    meta = {"version": "1.0.0", "stats": {}}
    assert meta.get("embedding_model") is None


def test_no_reembed_when_matching() -> None:
    meta = {"embedding_model": _DEFAULT_MODEL, "embedding_dimensions": 384}
    assert meta.get("embedding_model") == _DEFAULT_MODEL
    assert meta.get("embedding_dimensions") == 384


def test_ensure_current_embeddings_reembeds_and_updates_meta(tmp_path) -> None:
    get_embedding_dimensions.cache_clear()
    repo_path = tmp_path
    axon_dir = repo_path / ".axon"
    axon_dir.mkdir()
    meta_path = axon_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dimensions": 384})
        + "\n",
        encoding="utf-8",
    )

    storage = MagicMock()
    storage.load_graph.return_value = object()

    with patch("axon.core.ingestion.watcher.embed_graph", return_value=[MagicMock()]):
        migrated = ensure_current_embeddings(storage, repo_path)

    assert migrated is True
    storage.load_graph.assert_called_once_with()
    storage.store_embeddings.assert_called_once()
    updated_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert updated_meta["embedding_model"] == get_effective_embedding_model_name()
    assert updated_meta["embedding_dimensions"] == 384


def test_ensure_current_embeddings_reembeds_when_dimensions_change(tmp_path) -> None:
    get_embedding_dimensions.cache_clear()
    repo_path = tmp_path
    axon_dir = repo_path / ".axon"
    axon_dir.mkdir()
    (axon_dir / "meta.json").write_text(
        json.dumps({"embedding_model": _DEFAULT_MODEL, "embedding_dimensions": 384}) + "\n",
        encoding="utf-8",
    )

    storage = MagicMock()
    storage.load_graph.return_value = object()

    with patch.dict("os.environ", {"AXON_EMBEDDING_DIMENSIONS": "1024"}, clear=False):
        get_embedding_dimensions.cache_clear()
        with patch("axon.core.ingestion.watcher.embed_graph", return_value=[MagicMock()]):
            migrated = ensure_current_embeddings(storage, repo_path)

    assert migrated is True
    storage.load_graph.assert_called_once_with()
    storage.store_embeddings.assert_called_once()


def test_ensure_current_embeddings_noop_when_model_matches(tmp_path) -> None:
    get_embedding_dimensions.cache_clear()
    repo_path = tmp_path
    axon_dir = repo_path / ".axon"
    axon_dir.mkdir()
    (axon_dir / "meta.json").write_text(
        json.dumps({"embedding_model": _DEFAULT_MODEL, "embedding_dimensions": 384}) + "\n",
        encoding="utf-8",
    )

    storage = MagicMock()

    migrated = ensure_current_embeddings(storage, repo_path)

    assert migrated is False
    storage.load_graph.assert_not_called()
    storage.store_embeddings.assert_not_called()
