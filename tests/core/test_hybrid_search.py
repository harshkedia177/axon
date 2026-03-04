"""Tests for hybrid search combining FTS and vector search via RRF."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from axon.core.search.hybrid import hybrid_search
from axon.core.storage.base import SearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_storage() -> MagicMock:
    """Return a mock StorageBackend with FTS and vector results that overlap."""
    storage = MagicMock()
    # FTS returns results in ranked order
    storage.fts_search.return_value = [
        SearchResult(node_id="a", score=1.0, node_name="validate_user", file_path="src/auth.py", label="function"),
        SearchResult(node_id="b", score=0.8, node_name="validate_input", file_path="src/forms.py", label="function"),
        SearchResult(node_id="c", score=0.5, node_name="check_valid", file_path="src/utils.py", label="function"),
    ]
    # Vector returns results (some overlap with FTS)
    storage.vector_search.return_value = [
        SearchResult(node_id="b", score=0.95, node_name="validate_input", file_path="src/forms.py", label="function"),
        SearchResult(node_id="d", score=0.9, node_name="verify_user", file_path="src/verify.py", label="function"),
        SearchResult(node_id="a", score=0.7, node_name="validate_user", file_path="src/auth.py", label="function"),
    ]
    return storage


# ---------------------------------------------------------------------------
# Basic behaviour tests
# ---------------------------------------------------------------------------


class TestHybridSearchBasic:
    """Core hybrid search behaviour."""

    def test_returns_results_from_both_sources(self, mock_storage: MagicMock) -> None:
        """Results from both FTS and vector search should appear."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1, 0.2])
        node_ids = {r.node_id for r in results}
        # All four unique IDs should be present
        assert node_ids == {"a", "b", "c", "d"}

    def test_overlapping_items_boosted_to_top(self, mock_storage: MagicMock) -> None:
        """Items appearing in both ranked lists should be boosted above single-list items."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1, 0.2])
        ids = [r.node_id for r in results]
        # "a" and "b" appear in both lists, so they should be the top two
        assert set(ids[:2]) == {"a", "b"}

    def test_fts_only_when_no_embedding(self, mock_storage: MagicMock) -> None:
        """When query_embedding is None, only FTS results are returned."""
        results = hybrid_search("validate", mock_storage, query_embedding=None)
        assert len(results) == 3
        mock_storage.vector_search.assert_not_called()

    def test_limit_respected(self, mock_storage: MagicMock) -> None:
        """Output should not exceed the requested limit."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1], limit=2)
        assert len(results) <= 2

    def test_empty_results(self) -> None:
        """Both sources returning empty should produce an empty list."""
        storage = MagicMock()
        storage.fts_search.return_value = []
        storage.vector_search.return_value = []
        results = hybrid_search("nothing", storage, query_embedding=[0.1])
        assert results == []


# ---------------------------------------------------------------------------
# RRF scoring tests
# ---------------------------------------------------------------------------


class TestRRFScoring:
    """Verify the Reciprocal Rank Fusion score calculation."""

    def test_scores_are_positive(self, mock_storage: MagicMock) -> None:
        """Every returned result must have a positive RRF score."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1])
        for r in results:
            assert r.score > 0

    def test_dual_list_item_scores_higher_than_single(self, mock_storage: MagicMock) -> None:
        """An item in both lists should score strictly higher than one in only one list."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1])
        score_map = {r.node_id: r.score for r in results}
        # "c" only appears in FTS; "d" only in vector; "a" and "b" in both
        assert score_map["a"] > score_map["c"]
        assert score_map["b"] > score_map["d"]

    def test_rrf_scores_match_formula(self, mock_storage: MagicMock) -> None:
        """Manually verify RRF computation for known ranks with default k=60."""
        k = 60
        results = hybrid_search(
            "validate", mock_storage, query_embedding=[0.1],
            fts_weight=1.0, vector_weight=1.0, rrf_k=k,
        )
        score_map = {r.node_id: r.score for r in results}

        # "a": FTS rank 1, vector rank 3  ->  1/(60+1) + 1/(60+3)
        expected_a = 1.0 / (k + 1) + 1.0 / (k + 3)
        assert score_map["a"] == pytest.approx(expected_a)

        # "b": FTS rank 2, vector rank 1  ->  1/(60+2) + 1/(60+1)
        expected_b = 1.0 / (k + 2) + 1.0 / (k + 1)
        assert score_map["b"] == pytest.approx(expected_b)

        # "c": FTS rank 3 only  ->  1/(60+3)
        expected_c = 1.0 / (k + 3)
        assert score_map["c"] == pytest.approx(expected_c)

        # "d": vector rank 2 only  ->  1/(60+2)
        expected_d = 1.0 / (k + 2)
        assert score_map["d"] == pytest.approx(expected_d)

    def test_weights_affect_scores(self, mock_storage: MagicMock) -> None:
        """Adjusting fts_weight and vector_weight should change resulting scores."""
        k = 60
        results_fts_heavy = hybrid_search(
            "validate", mock_storage, query_embedding=[0.1],
            fts_weight=2.0, vector_weight=0.5, rrf_k=k,
        )
        fts_heavy_scores = {r.node_id: r.score for r in results_fts_heavy}

        results_vec_heavy = hybrid_search(
            "validate", mock_storage, query_embedding=[0.1],
            fts_weight=0.5, vector_weight=2.0, rrf_k=k,
        )
        vec_heavy_scores = {r.node_id: r.score for r in results_vec_heavy}

        # "c" only appears in FTS: fts_heavy should give higher score
        assert fts_heavy_scores["c"] > vec_heavy_scores["c"]
        # "d" only appears in vector: vec_heavy should give higher score
        assert vec_heavy_scores["d"] > fts_heavy_scores["d"]

    def test_custom_rrf_k(self, mock_storage: MagicMock) -> None:
        """Changing rrf_k should change the scores."""
        results_small_k = hybrid_search(
            "validate", mock_storage, query_embedding=[0.1], rrf_k=1,
        )
        results_large_k = hybrid_search(
            "validate", mock_storage, query_embedding=[0.1], rrf_k=100,
        )
        # With a smaller k, scores are larger (denominator smaller)
        small_k_top = results_small_k[0].score
        large_k_top = results_large_k[0].score
        assert small_k_top > large_k_top


# ---------------------------------------------------------------------------
# Result ordering tests
# ---------------------------------------------------------------------------


class TestResultOrdering:
    """Ensure results are sorted correctly by RRF score."""

    def test_descending_score_order(self, mock_storage: MagicMock) -> None:
        """Results should be sorted by descending RRF score."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_fts_only_preserves_order(self, mock_storage: MagicMock) -> None:
        """FTS-only mode should also sort by descending RRF score."""
        results = hybrid_search("validate", mock_storage, query_embedding=None)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Metadata preservation tests
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    """Hybrid search should carry through metadata from source results."""

    def test_node_name_preserved(self, mock_storage: MagicMock) -> None:
        """node_name should be populated from the original results."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1])
        names = {r.node_name for r in results}
        assert "validate_user" in names
        assert "validate_input" in names
        assert "verify_user" in names

    def test_file_path_preserved(self, mock_storage: MagicMock) -> None:
        """file_path should be populated from the original results."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1])
        paths = {r.file_path for r in results}
        assert "src/auth.py" in paths
        assert "src/forms.py" in paths

    def test_label_preserved(self, mock_storage: MagicMock) -> None:
        """label field should be preserved."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1])
        for r in results:
            assert r.label == "function"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_vector_only_results(self) -> None:
        """When FTS returns nothing but vector returns results, those should appear."""
        storage = MagicMock()
        storage.fts_search.return_value = []
        storage.vector_search.return_value = [
            SearchResult(node_id="x", score=0.9, node_name="find_me"),
        ]
        results = hybrid_search("query", storage, query_embedding=[0.1])
        assert len(results) == 1
        assert results[0].node_id == "x"

    def test_fts_only_results_no_vector(self) -> None:
        """When vector returns nothing but FTS returns results, those should appear."""
        storage = MagicMock()
        storage.fts_search.return_value = [
            SearchResult(node_id="y", score=0.8, node_name="keyword_hit"),
        ]
        storage.vector_search.return_value = []
        results = hybrid_search("query", storage, query_embedding=[0.5])
        assert len(results) == 1
        assert results[0].node_id == "y"

    def test_limit_zero_returns_empty(self, mock_storage: MagicMock) -> None:
        """Requesting limit=0 should return an empty list."""
        results = hybrid_search("validate", mock_storage, query_embedding=[0.1], limit=0)
        assert results == []

    def test_fts_called_with_expanded_limit(self, mock_storage: MagicMock) -> None:
        """FTS should be called with limit * 3 to gather enough candidates."""
        hybrid_search("validate", mock_storage, query_embedding=[0.1], limit=10)
        mock_storage.fts_search.assert_called_once_with("validate", limit=30)
        mock_storage.vector_search.assert_called_once_with([0.1], limit=30)

    def test_duplicate_node_id_in_same_list(self) -> None:
        """If a source returns duplicates, only the first occurrence rank matters."""
        storage = MagicMock()
        storage.fts_search.return_value = [
            SearchResult(node_id="dup", score=1.0, node_name="dup_func"),
            SearchResult(node_id="dup", score=0.5, node_name="dup_func"),
        ]
        storage.vector_search.return_value = []
        results = hybrid_search("dup", storage, query_embedding=[0.1])
        # Should only appear once
        assert sum(1 for r in results if r.node_id == "dup") == 1
