from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from axon.core.embeddings.embedder import (
    EMBEDDABLE_LABELS,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_DIMENSIONS,
    _DEFAULT_MODEL,
    _get_model,
    configure_cuda,
    embed_graph,
    embed_nodes,
    embed_query,
    validate_cuda,
)
from axon.core.embeddings.embedder import _resolve_cuda
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType, generate_id
from axon.core.storage.base import EMBEDDING_DIMENSIONS, NodeEmbedding


def _vec768(base: list[float] | None = None) -> np.ndarray:
    """Return a 768-d numpy array (nomic native dim), zero-padded from *base*."""
    v = np.zeros(768)
    if base:
        v[: len(base)] = base
    return v


@pytest.fixture(autouse=True)
def _clear_model_cache(monkeypatch):
    """Reset embedding state before each test so mocks work."""
    _get_model.cache_clear()
    configure_cuda(False)
    monkeypatch.delenv("AXON_CUDA", raising=False)
    yield
    _get_model.cache_clear()


@pytest.fixture
def sample_graph() -> KnowledgeGraph:
    """Graph with two embeddable nodes (function, class) and one non-embeddable (folder)."""
    graph = KnowledgeGraph()
    graph.add_node(
        GraphNode(
            id="function:src/a.py:foo",
            label=NodeLabel.FUNCTION,
            name="foo",
            file_path="src/a.py",
        )
    )
    graph.add_node(
        GraphNode(
            id="class:src/a.py:Bar",
            label=NodeLabel.CLASS,
            name="Bar",
            file_path="src/a.py",
        )
    )
    graph.add_node(
        GraphNode(
            id="folder::src",
            label=NodeLabel.FOLDER,
            name="src",
        )
    )
    return graph


@pytest.fixture
def all_label_graph() -> KnowledgeGraph:
    """Graph containing one node of every label for completeness testing."""
    graph = KnowledgeGraph()
    nodes = [
        GraphNode(id="file:src/a.py:", label=NodeLabel.FILE, name="a.py", file_path="src/a.py"),
        GraphNode(
            id="function:src/a.py:foo",
            label=NodeLabel.FUNCTION,
            name="foo",
            file_path="src/a.py",
        ),
        GraphNode(
            id="class:src/a.py:Bar",
            label=NodeLabel.CLASS,
            name="Bar",
            file_path="src/a.py",
        ),
        GraphNode(
            id="method:src/a.py:baz",
            label=NodeLabel.METHOD,
            name="baz",
            file_path="src/a.py",
            class_name="Bar",
        ),
        GraphNode(
            id="interface:src/types.ts:IFoo",
            label=NodeLabel.INTERFACE,
            name="IFoo",
            file_path="src/types.ts",
        ),
        GraphNode(
            id="type_alias:src/types.py:UserID",
            label=NodeLabel.TYPE_ALIAS,
            name="UserID",
            file_path="src/types.py",
        ),
        GraphNode(
            id="enum:src/enums.py:Color",
            label=NodeLabel.ENUM,
            name="Color",
            file_path="src/enums.py",
        ),
        # Non-embeddable labels:
        GraphNode(id="folder::src", label=NodeLabel.FOLDER, name="src"),
        GraphNode(id="community::auth", label=NodeLabel.COMMUNITY, name="auth"),
        GraphNode(id="process::login", label=NodeLabel.PROCESS, name="login"),
    ]
    for n in nodes:
        graph.add_node(n)
    return graph


class TestModelDefaults:
    def test_default_model_is_nomic(self) -> None:
        assert "nomic" in _DEFAULT_MODEL

    def test_default_dimensions(self) -> None:
        assert _DEFAULT_DIMENSIONS == 384

    def test_default_batch_size(self) -> None:
        assert _DEFAULT_BATCH_SIZE == 8


class TestEmbeddableLabels:
    def test_contains_expected_labels(self) -> None:
        expected = {
            NodeLabel.FILE,
            NodeLabel.FUNCTION,
            NodeLabel.CLASS,
            NodeLabel.METHOD,
            NodeLabel.INTERFACE,
            NodeLabel.TYPE_ALIAS,
            NodeLabel.ENUM,
        }
        assert EMBEDDABLE_LABELS == expected

    def test_excludes_structural_labels(self) -> None:
        assert NodeLabel.FOLDER not in EMBEDDABLE_LABELS
        assert NodeLabel.COMMUNITY not in EMBEDDABLE_LABELS
        assert NodeLabel.PROCESS not in EMBEDDABLE_LABELS

    def test_is_frozenset(self) -> None:
        assert isinstance(EMBEDDABLE_LABELS, frozenset)


class TestEmbedGraphBasic:
    @patch("fastembed.TextEmbedding")
    def test_returns_node_embeddings(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        assert len(results) == 2  # function + class; folder is skipped
        assert all(isinstance(r, NodeEmbedding) for r in results)

    @patch("fastembed.TextEmbedding")
    def test_embedding_vectors_are_lists_of_float(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        for r in results:
            assert isinstance(r.embedding, list)
            assert all(isinstance(v, float) for v in r.embedding)

    @patch("fastembed.TextEmbedding")
    def test_embedding_values_match_after_truncation(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        # After Matryoshka truncation to 384 dims, values should match first 384 of 768
        embeddings = [r.embedding for r in results]
        vec_a = _vec768([0.1, 0.2, 0.3])[:384].tolist()
        vec_b = _vec768([0.4, 0.5, 0.6])[:384].tolist()
        assert vec_a in embeddings or pytest.approx(vec_a) in embeddings
        assert vec_b in embeddings or pytest.approx(vec_b) in embeddings

    @patch("fastembed.TextEmbedding")
    def test_node_ids_are_correct(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        node_ids = {r.node_id for r in results}
        assert "function:src/a.py:foo" in node_ids
        assert "class:src/a.py:Bar" in node_ids


class TestEmbedGraphFiltering:
    @patch("fastembed.TextEmbedding")
    def test_skips_folder_nodes(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        node_ids = {r.node_id for r in results}
        assert "folder::src" not in node_ids

    @patch("fastembed.TextEmbedding")
    def test_skips_community_and_process(
        self, mock_te_cls: MagicMock, all_label_graph: KnowledgeGraph
    ) -> None:
        embeddable_count = 7  # FILE, FUNCTION, CLASS, METHOD, INTERFACE, TYPE_ALIAS, ENUM
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]) for _ in range(embeddable_count)]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(all_label_graph)

        assert len(results) == embeddable_count
        node_ids = {r.node_id for r in results}
        assert "folder::src" not in node_ids
        assert "community::auth" not in node_ids
        assert "process::login" not in node_ids

    @patch("fastembed.TextEmbedding")
    def test_all_embeddable_labels_included(
        self, mock_te_cls: MagicMock, all_label_graph: KnowledgeGraph
    ) -> None:
        embeddable_count = 7
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]) for _ in range(embeddable_count)]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(all_label_graph)

        node_ids = {r.node_id for r in results}
        assert "file:src/a.py:" in node_ids
        assert "function:src/a.py:foo" in node_ids
        assert "class:src/a.py:Bar" in node_ids
        assert "method:src/a.py:baz" in node_ids
        assert "interface:src/types.ts:IFoo" in node_ids
        assert "type_alias:src/types.py:UserID" in node_ids
        assert "enum:src/enums.py:Color" in node_ids


class TestEmbedGraphEmpty:
    @patch("fastembed.TextEmbedding")
    def test_empty_graph_returns_empty_list(self, mock_te_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter([])
        mock_te_cls.return_value = mock_model

        graph = KnowledgeGraph()
        results = embed_graph(graph)

        assert results == []

    @patch("fastembed.TextEmbedding")
    def test_graph_with_only_non_embeddable_returns_empty(self, mock_te_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter([])
        mock_te_cls.return_value = mock_model

        graph = KnowledgeGraph()
        graph.add_node(
            GraphNode(id="folder::src", label=NodeLabel.FOLDER, name="src")
        )
        graph.add_node(
            GraphNode(id="community::auth", label=NodeLabel.COMMUNITY, name="auth")
        )

        results = embed_graph(graph)

        assert results == []


class TestEmbedGraphModelConfig:
    @patch("fastembed.TextEmbedding")
    def test_default_model_name(self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        mock_te_cls.assert_called_once()
        assert mock_te_cls.call_args.kwargs["model_name"] == "nomic-ai/nomic-embed-text-v1.5"
        assert mock_te_cls.call_args.kwargs["threads"] >= 2

    @patch("fastembed.TextEmbedding")
    def test_custom_model_name(self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph, model_name="BAAI/bge-base-en-v1.5")

        mock_te_cls.assert_called_once()
        assert mock_te_cls.call_args.kwargs["model_name"] == "BAAI/bge-base-en-v1.5"

    @patch("fastembed.TextEmbedding")
    def test_custom_batch_size_passed_to_passage_embed(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph, batch_size=32)

        # Verify batch_size was passed to passage_embed()
        embed_call = mock_model.passage_embed.call_args
        assert embed_call.kwargs.get("batch_size") == 32 or (
            len(embed_call.args) > 1 and embed_call.args[1] == 32
        )


class TestEmbedGraphTextGeneration:
    @patch("axon.core.embeddings.embedder.generate_text")
    @patch("fastembed.TextEmbedding")
    def test_generate_text_called_for_each_node(
        self,
        mock_te_cls: MagicMock,
        mock_gen_text: MagicMock,
        sample_graph: KnowledgeGraph,
    ) -> None:
        mock_gen_text.return_value = "mock text"
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        # generate_text should be called twice (function + class, not folder)
        assert mock_gen_text.call_count == 2

    @patch("axon.core.embeddings.embedder.generate_text")
    @patch("fastembed.TextEmbedding")
    def test_generated_texts_passed_to_model(
        self,
        mock_te_cls: MagicMock,
        mock_gen_text: MagicMock,
        sample_graph: KnowledgeGraph,
    ) -> None:
        mock_gen_text.side_effect = ["text for foo", "text for Bar"]
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        # The texts list passed to model.passage_embed should contain both texts
        embed_call_args = mock_model.passage_embed.call_args
        texts_arg = (
            embed_call_args.args[0] if embed_call_args.args
            else embed_call_args.kwargs.get("documents", [])
        )
        assert "text for foo" in texts_arg
        assert "text for Bar" in texts_arg


class TestEmbedGraphBatchProcessing:
    @patch("fastembed.TextEmbedding")
    def test_many_nodes_all_embedded(self, mock_te_cls: MagicMock) -> None:
        graph = KnowledgeGraph()
        count = 100
        for i in range(count):
            graph.add_node(
                GraphNode(
                    id=f"function:src/mod.py:fn_{i}",
                    label=NodeLabel.FUNCTION,
                    name=f"fn_{i}",
                    file_path="src/mod.py",
                )
            )

        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([float(i), float(i + 1), float(i + 2)]) for i in range(count)]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(graph, batch_size=16)

        assert len(results) == count
        # Each embedding should have 384 dimensions (Matryoshka truncation)
        assert all(len(r.embedding) == EMBEDDING_DIMENSIONS for r in results)

    @patch("fastembed.TextEmbedding")
    def test_default_batch_size_is_8(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        embed_call = mock_model.passage_embed.call_args
        assert embed_call.kwargs.get("batch_size") == 8 or (
            len(embed_call.args) > 1 and embed_call.args[1] == 8
        )


class TestEmbedGraphNomic:
    @patch("fastembed.TextEmbedding")
    def test_uses_passage_embed(self, mock_te_cls, sample_graph):
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter([_vec768([0.1]), _vec768([0.2])])
        mock_te_cls.return_value = mock_model
        results = embed_graph(sample_graph)
        mock_model.passage_embed.assert_called_once()
        mock_model.embed.assert_not_called()

    @patch("fastembed.TextEmbedding")
    def test_matryoshka_truncation(self, mock_te_cls, sample_graph):
        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter([_vec768([0.1]), _vec768([0.2])])
        mock_te_cls.return_value = mock_model
        results = embed_graph(sample_graph)
        for r in results:
            assert len(r.embedding) == 384


class TestEmbedQueryNomic:
    @patch("fastembed.TextEmbedding")
    def test_uses_query_embed(self, mock_te_cls):
        mock_model = MagicMock()
        mock_model.query_embed.return_value = iter([np.array([0.5] * 768)])
        mock_te_cls.return_value = mock_model
        result = embed_query("test query")
        mock_model.query_embed.assert_called_once()
        mock_model.embed.assert_not_called()

    @patch("fastembed.TextEmbedding")
    def test_query_embed_truncates_to_384(self, mock_te_cls):
        mock_model = MagicMock()
        mock_model.query_embed.return_value = iter([np.array([0.5] * 768)])
        mock_te_cls.return_value = mock_model
        result = embed_query("test query")
        assert result is not None
        assert len(result) == 384


def _make_incremental_graph() -> KnowledgeGraph:
    """Build a small graph with two functions that call each other."""
    graph = KnowledgeGraph()
    fn_a = GraphNode(
        id=generate_id(NodeLabel.FUNCTION, "src/a.py", "func_a"),
        label=NodeLabel.FUNCTION, name="func_a", file_path="src/a.py",
        signature="def func_a():",
    )
    fn_b = GraphNode(
        id=generate_id(NodeLabel.FUNCTION, "src/b.py", "func_b"),
        label=NodeLabel.FUNCTION, name="func_b", file_path="src/b.py",
        signature="def func_b():",
    )
    graph.add_node(fn_a)
    graph.add_node(fn_b)
    graph.add_relationship(GraphRelationship(
        id=f"calls:{fn_a.id}->{fn_b.id}",
        type=RelType.CALLS, source=fn_a.id, target=fn_b.id,
    ))
    return graph


class TestEmbeddingAlignment:
    @patch("axon.core.embeddings.embedder.generate_text")
    @patch("fastembed.TextEmbedding")
    def test_embedding_alignment(
        self, mock_te_cls: MagicMock, mock_gen_text: MagicMock
    ) -> None:
        graph = KnowledgeGraph()
        node_a = GraphNode(
            id="function:src/a.py:func_a",
            label=NodeLabel.FUNCTION,
            name="func_a",
            file_path="src/a.py",
        )
        node_b = GraphNode(
            id="function:src/b.py:func_b",
            label=NodeLabel.FUNCTION,
            name="func_b",
            file_path="src/b.py",
        )
        graph.add_node(node_a)
        graph.add_node(node_b)

        # Distinguishable texts so we know which node maps to which vector.
        mock_gen_text.side_effect = lambda node, *args, **kwargs: (
            "text for func_a" if node.id == node_a.id else "text for func_b"
        )

        # Two distinguishable embedding vectors (768d, truncated to 384).
        embedding_a = _vec768([1.0, 0.0, 0.0])
        embedding_b = _vec768([0.0, 1.0, 0.0])

        mock_model = MagicMock()
        mock_te_cls.return_value = mock_model
        # The model yields vectors in the same order texts are passed.
        mock_model.passage_embed.return_value = iter([embedding_a, embedding_b])

        results = embed_graph(graph)

        assert len(results) == 2
        by_id = {r.node_id: r.embedding for r in results}
        assert node_a.id in by_id
        assert node_b.id in by_id
        # The two nodes must have received different embeddings.
        assert by_id[node_a.id] != by_id[node_b.id]


class TestEmbedNodes:
    @patch("fastembed.TextEmbedding")
    def test_embeds_only_requested_nodes(self, mock_te_cls: MagicMock) -> None:
        graph = _make_incremental_graph()
        node_ids = {generate_id(NodeLabel.FUNCTION, "src/a.py", "func_a")}

        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter([_vec768([0.1, 0.2, 0.3])])
        mock_te_cls.return_value = mock_model

        results = embed_nodes(graph, node_ids)

        result_ids = {emb.node_id for emb in results}
        assert node_ids == result_ids

    def test_returns_empty_for_empty_set(self) -> None:
        graph = _make_incremental_graph()
        results = embed_nodes(graph, set())
        assert results == []

    @patch("fastembed.TextEmbedding")
    def test_skips_non_embeddable_labels(self, mock_te_cls: MagicMock) -> None:
        graph = KnowledgeGraph()
        folder = GraphNode(
            id=generate_id(NodeLabel.FOLDER, "src", "src"),
            label=NodeLabel.FOLDER, name="src", file_path="src",
        )
        graph.add_node(folder)
        results = embed_nodes(graph, {folder.id})
        assert results == []

    @patch("fastembed.TextEmbedding")
    def test_skips_missing_node_ids(self, mock_te_cls: MagicMock) -> None:
        graph = _make_incremental_graph()
        results = embed_nodes(graph, {"function:nonexistent.py:nope"})
        assert results == []

    @patch("fastembed.TextEmbedding")
    def test_embeds_both_requested_nodes(self, mock_te_cls: MagicMock) -> None:
        graph = _make_incremental_graph()
        id_a = generate_id(NodeLabel.FUNCTION, "src/a.py", "func_a")
        id_b = generate_id(NodeLabel.FUNCTION, "src/b.py", "func_b")
        node_ids = {id_a, id_b}

        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter(
            [_vec768([0.1, 0.2, 0.3]), _vec768([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_nodes(graph, node_ids)

        result_ids = {emb.node_id for emb in results}
        assert result_ids == node_ids
        assert len(results) == 2

    @patch("fastembed.TextEmbedding")
    def test_embedding_values_are_correct(self, mock_te_cls: MagicMock) -> None:
        graph = _make_incremental_graph()
        id_a = generate_id(NodeLabel.FUNCTION, "src/a.py", "func_a")

        mock_model = MagicMock()
        mock_model.passage_embed.return_value = iter([_vec768([1.0, 2.0, 3.0])])
        mock_te_cls.return_value = mock_model

        results = embed_nodes(graph, {id_a})

        assert len(results) == 1
        assert results[0].node_id == id_a
        assert isinstance(results[0].embedding, list)
        # After Matryoshka truncation, we get first 384 dims of the 768d vector
        expected = _vec768([1.0, 2.0, 3.0])[:384].tolist()
        assert results[0].embedding == pytest.approx(expected)


class TestCudaSupport:
    """CUDA support flag and model initialization tests."""

    def test_cuda_disabled_by_default(self) -> None:
        """_resolve_cuda returns False with no flag set and no env var."""
        assert _resolve_cuda() is False

    def test_configure_cuda_sets_flag(self) -> None:
        """configure_cuda(True) causes _resolve_cuda to return True."""
        try:
            configure_cuda(True)
            assert _resolve_cuda() is True
        finally:
            configure_cuda(False)

    def test_axon_cuda_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AXON_CUDA=1 env var enables CUDA without calling configure_cuda."""
        monkeypatch.setenv("AXON_CUDA", "1")
        assert _resolve_cuda() is True

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("0", False),
            ("false", False),
            ("", False),
        ],
    )
    def test_axon_cuda_env_var_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
        expected: bool,
    ) -> None:
        """AXON_CUDA env var is recognized only for truthy values."""
        monkeypatch.setenv("AXON_CUDA", value)
        assert _resolve_cuda() is expected

    @patch("fastembed.TextEmbedding")
    def test_get_model_passes_cuda_to_text_embedding(
        self, mock_te_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TextEmbedding is instantiated with cuda=True when AXON_CUDA=1."""
        monkeypatch.setenv("AXON_CUDA", "1")
        mock_te_cls.return_value = MagicMock()

        _get_model("test-model")

        _, kwargs = mock_te_cls.call_args
        assert kwargs.get("cuda") is True

    @patch("fastembed.TextEmbedding")
    def test_get_model_no_cuda_by_default(
        self, mock_te_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TextEmbedding is instantiated with cuda=False to override auto-detect."""
        monkeypatch.delenv("AXON_CUDA", raising=False)
        mock_te_cls.return_value = MagicMock()

        _get_model("test-model")

        _, kwargs = mock_te_cls.call_args
        assert kwargs.get("cuda") is False

    @patch("fastembed.TextEmbedding")
    def test_cuda_cache_key_separation(
        self, mock_te_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CUDA and non-CUDA models are cached under separate keys."""
        monkeypatch.delenv("AXON_CUDA", raising=False)
        mock_te_cls.return_value = MagicMock()

        _get_model("test-model")
        try:
            configure_cuda(True)
            _get_model("test-model")
        finally:
            configure_cuda(False)

        assert mock_te_cls.call_count == 2

    @patch("fastembed.TextEmbedding")
    def test_cuda_fallback_raises_runtime_error(
        self, mock_te_cls: MagicMock
    ) -> None:
        """RuntimeWarning with CUDAExecutionProvider is re-raised as RuntimeError."""
        def _emit_cuda_warning(*args: object, **kwargs: object) -> MagicMock:
            warnings.warn(
                "CUDAExecutionProvider not available", RuntimeWarning
            )
            return MagicMock()

        mock_te_cls.side_effect = _emit_cuda_warning
        try:
            configure_cuda(True)
            with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
                _get_model("test-model")
        finally:
            configure_cuda(False)

    @patch("fastembed.TextEmbedding")
    def test_validate_cuda_noop_when_disabled(
        self, mock_te_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """validate_cuda does not load the model when CUDA is disabled."""
        monkeypatch.delenv("AXON_CUDA", raising=False)

        validate_cuda()

        mock_te_cls.assert_not_called()

    @patch("fastembed.TextEmbedding")
    def test_validate_cuda_raises_on_fallback(
        self, mock_te_cls: MagicMock
    ) -> None:
        """validate_cuda propagates RuntimeError when CUDA provider fails."""
        def _emit_cuda_warning(*args: object, **kwargs: object) -> MagicMock:
            warnings.warn(
                "CUDAExecutionProvider not available", RuntimeWarning
            )
            return MagicMock()

        mock_te_cls.side_effect = _emit_cuda_warning
        try:
            configure_cuda(True)
            with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
                validate_cuda()
        finally:
            configure_cuda(False)
