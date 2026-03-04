"""Tests for Axon MCP tool handlers.

All tests mock the storage backend to avoid needing a real database.
Each tool handler is tested for both success and edge-case paths.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from axon.core.graph.model import GraphNode, NodeLabel
from axon.core.storage.base import SearchResult
from axon.mcp.tools import (
    _confidence_tag,
    _format_query_results,
    _group_by_process,
    handle_context,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_impact,
    handle_list_repos,
    handle_query,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    """Create a mock storage backend with common default return values."""
    storage = MagicMock()
    storage.fts_search.return_value = [
        SearchResult(
            node_id="function:src/auth.py:validate",
            score=1.0,
            node_name="validate",
            file_path="src/auth.py",
            label="function",
            snippet="def validate(user): ...",
        ),
    ]
    storage.get_node.return_value = GraphNode(
        id="function:src/auth.py:validate",
        label=NodeLabel.FUNCTION,
        name="validate",
        file_path="src/auth.py",
        start_line=10,
        end_line=30,
    )
    storage.get_callers.return_value = []
    storage.get_callees.return_value = []
    storage.get_type_refs.return_value = []
    storage.vector_search.return_value = []
    storage.traverse.return_value = []
    storage.traverse_with_depth.return_value = []
    storage.get_callers_with_confidence.return_value = []
    storage.get_callees_with_confidence.return_value = []
    storage.get_process_memberships.return_value = {}
    storage.execute_raw.return_value = []
    return storage


@pytest.fixture
def mock_storage_with_relations(mock_storage):
    """Storage mock with callers, callees, and type refs populated."""
    _caller = GraphNode(
        id="function:src/routes/auth.py:login_handler",
        label=NodeLabel.FUNCTION,
        name="login_handler",
        file_path="src/routes/auth.py",
        start_line=12,
        end_line=40,
    )
    _callee = GraphNode(
        id="function:src/auth/crypto.py:hash_password",
        label=NodeLabel.FUNCTION,
        name="hash_password",
        file_path="src/auth/crypto.py",
        start_line=5,
        end_line=20,
    )
    mock_storage.get_callers.return_value = [_caller]
    mock_storage.get_callees.return_value = [_callee]
    mock_storage.get_callers_with_confidence.return_value = [(_caller, 1.0)]
    mock_storage.get_callees_with_confidence.return_value = [(_callee, 0.8)]
    mock_storage.get_type_refs.return_value = [
        GraphNode(
            id="class:src/models.py:User",
            label=NodeLabel.CLASS,
            name="User",
            file_path="src/models.py",
            start_line=1,
            end_line=50,
        ),
    ]
    return mock_storage


# ---------------------------------------------------------------------------
# 1. axon_list_repos
# ---------------------------------------------------------------------------


class TestHandleListRepos:
    def test_no_registry_dir(self, tmp_path):
        """Returns 'no repos' message when registry directory does not exist."""
        result = handle_list_repos(registry_dir=tmp_path / "nonexistent")
        assert "No indexed repositories found" in result

    def test_empty_registry_dir(self, tmp_path):
        """Returns 'no repos' message when registry directory is empty."""
        registry = tmp_path / "repos"
        registry.mkdir()
        result = handle_list_repos(registry_dir=registry)
        assert "No indexed repositories found" in result

    def test_with_repos(self, tmp_path):
        """Returns formatted repo list when meta.json files are present."""
        registry = tmp_path / "repos"
        repo_dir = registry / "my-project"
        repo_dir.mkdir(parents=True)
        meta = {
            "name": "my-project",
            "path": "/home/user/my-project",
            "stats": {
                "files": 25,
                "symbols": 150,
                "relationships": 200,
            },
        }
        (repo_dir / "meta.json").write_text(json.dumps(meta))

        result = handle_list_repos(registry_dir=registry)
        assert "my-project" in result
        assert "150" in result
        assert "200" in result
        assert "Indexed repositories (1)" in result


# ---------------------------------------------------------------------------
# 2. axon_query
# ---------------------------------------------------------------------------


class TestHandleQuery:
    def test_returns_results(self, mock_storage):
        """Successful query returns formatted results."""
        result = handle_query(mock_storage, "validate")
        assert "validate" in result
        assert "Function" in result
        assert "src/auth.py" in result
        assert "Next:" in result

    def test_no_results(self, mock_storage):
        """Empty search returns no-results message."""
        mock_storage.fts_search.return_value = []
        mock_storage.vector_search.return_value = []
        result = handle_query(mock_storage, "nonexistent")
        assert "No results found" in result

    def test_snippet_included(self, mock_storage):
        """Search results include snippet text."""
        result = handle_query(mock_storage, "validate")
        assert "def validate" in result

    def test_custom_limit(self, mock_storage):
        """Limit parameter is passed through to hybrid_search."""
        handle_query(mock_storage, "validate", limit=5)
        # hybrid_search calls fts_search with candidate_limit = limit * 3
        mock_storage.fts_search.assert_called_once_with("validate", limit=15)


# ---------------------------------------------------------------------------
# 3. axon_context
# ---------------------------------------------------------------------------


class TestHandleContext:
    def test_basic_context(self, mock_storage):
        """Returns symbol name, file, and line range."""
        result = handle_context(mock_storage, "validate")
        assert "Symbol: validate (Function)" in result
        assert "src/auth.py:10-30" in result
        assert "Next:" in result

    def test_not_found_fts_empty(self, mock_storage):
        """Returns not-found message when FTS returns nothing."""
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_context(mock_storage, "nonexistent")
        assert "not found" in result.lower()

    def test_not_found_node_none(self, mock_storage):
        """Returns not-found message when get_node returns None."""
        mock_storage.get_node.return_value = None
        result = handle_context(mock_storage, "validate")
        assert "not found" in result.lower()

    def test_with_callers_callees_type_refs(self, mock_storage_with_relations):
        """Full context includes callers, callees, and type refs."""
        result = handle_context(mock_storage_with_relations, "validate")
        assert "Callers (1):" in result
        assert "login_handler" in result
        assert "Callees (1):" in result
        assert "hash_password" in result
        assert "Type references (1):" in result
        assert "User" in result

    def test_dead_code_flag(self, mock_storage):
        """Dead code status is shown when is_dead is True."""
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/old.py:deprecated",
            label=NodeLabel.FUNCTION,
            name="deprecated",
            file_path="src/old.py",
            start_line=1,
            end_line=5,
            is_dead=True,
        )
        result = handle_context(mock_storage, "deprecated")
        assert "DEAD CODE" in result


# ---------------------------------------------------------------------------
# 4. axon_impact
# ---------------------------------------------------------------------------


class TestHandleImpact:
    def test_no_downstream(self, mock_storage):
        """Returns no-dependencies message when traverse is empty."""
        result = handle_impact(mock_storage, "validate")
        assert "No upstream callers found" in result or "No downstream dependencies" in result

    def test_with_affected_symbols(self, mock_storage):
        """Returns formatted impact list when traverse finds nodes."""
        _login = GraphNode(
            id="function:src/api.py:login",
            label=NodeLabel.FUNCTION,
            name="login",
            file_path="src/api.py",
            start_line=5,
            end_line=20,
        )
        _register = GraphNode(
            id="function:src/api.py:register",
            label=NodeLabel.FUNCTION,
            name="register",
            file_path="src/api.py",
            start_line=25,
            end_line=50,
        )
        mock_storage.traverse.return_value = [_login, _register]
        mock_storage.traverse_with_depth.return_value = [(_login, 1), (_register, 2)]
        mock_storage.get_callers_with_confidence.return_value = [(_login, 1.0)]
        result = handle_impact(mock_storage, "validate", depth=2)
        assert "Impact analysis for: validate" in result
        assert "Total: 2 symbols" in result
        assert "login" in result
        assert "register" in result
        assert "Depth: 2" in result

    def test_symbol_not_found(self, mock_storage):
        """Returns not-found when symbol does not exist."""
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_impact(mock_storage, "nonexistent")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# 5. axon_dead_code
# ---------------------------------------------------------------------------


class TestHandleDeadCode:
    def test_no_dead_code(self, mock_storage):
        """Returns clean message when no dead code found."""
        result = handle_dead_code(mock_storage)
        assert "No dead code detected" in result

    def test_with_dead_code(self, mock_storage):
        """Returns formatted dead code list (delegates to get_dead_code_list)."""
        mock_storage.execute_raw.return_value = [
            ["unused_func", "src/old.py", 10],
            ["DeprecatedModel", "src/models.py", 5],
        ]
        result = handle_dead_code(mock_storage)
        assert "Dead Code Report (2 symbols)" in result
        assert "unused_func" in result
        assert "DeprecatedModel" in result

    def test_execute_raw_exception(self, mock_storage):
        """Gracefully handles storage errors."""
        mock_storage.execute_raw.side_effect = RuntimeError("DB error")
        result = handle_dead_code(mock_storage)
        assert "Could not retrieve dead code list" in result


# ---------------------------------------------------------------------------
# 6. axon_detect_changes
# ---------------------------------------------------------------------------


SAMPLE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index abc1234..def5678 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,5 +10,7 @@ def validate(user):
     if not user:
         return False
+    # Added new validation
+    check_permissions(user)
     return True
"""


class TestHandleDetectChanges:
    def test_parses_diff(self, mock_storage):
        """Successfully parses diff and identifies changed files."""
        # handle_detect_changes now uses execute_raw() with a Cypher query
        # to find symbols in the changed file.
        mock_storage.execute_raw.return_value = [
            ["function:src/auth.py:validate", "validate", "src/auth.py", 10, 30],
        ]

        result = handle_detect_changes(mock_storage, SAMPLE_DIFF)
        assert "src/auth.py" in result
        assert "validate" in result
        assert "Total affected symbols:" in result

    def test_empty_diff(self, mock_storage):
        """Returns message for empty diff input."""
        result = handle_detect_changes(mock_storage, "")
        assert "Empty diff provided" in result

    def test_unparseable_diff(self, mock_storage):
        """Returns message when diff contains no recognisable hunks."""
        result = handle_detect_changes(mock_storage, "just some random text")
        assert "Could not parse" in result

    def test_no_symbols_in_changed_lines(self, mock_storage):
        """Reports file but no symbols when nothing overlaps."""
        mock_storage.execute_raw.return_value = []
        result = handle_detect_changes(mock_storage, SAMPLE_DIFF)
        assert "src/auth.py" in result
        assert "no indexed symbols" in result


# ---------------------------------------------------------------------------
# 7. axon_cypher
# ---------------------------------------------------------------------------


class TestHandleCypher:
    def test_returns_results(self, mock_storage):
        """Formats raw query results."""
        mock_storage.execute_raw.return_value = [
            ["validate", "src/auth.py", 10],
            ["login", "src/api.py", 5],
        ]
        result = handle_cypher(mock_storage, "MATCH (n) RETURN n.name, n.file_path, n.start_line")
        assert "Results (2 rows)" in result
        assert "validate" in result
        assert "src/api.py" in result

    def test_no_results(self, mock_storage):
        """Returns no-results message for empty query output."""
        result = handle_cypher(mock_storage, "MATCH (n:Nonexistent) RETURN n")
        assert "no results" in result.lower()

    def test_query_error(self, mock_storage):
        """Returns error message when query execution fails."""
        mock_storage.execute_raw.side_effect = RuntimeError("Syntax error")
        result = handle_cypher(mock_storage, "INVALID QUERY")
        assert "failed" in result.lower()
        assert "Syntax error" in result

    def test_allows_read_only_call_query(self, mock_storage):
        """Allows read-only CALL procedures such as QUERY_FTS_INDEX."""
        query = "CALL QUERY_FTS_INDEX('Function', 'function_fts', 'validate') RETURN node.id, score"
        mock_storage.execute_raw.return_value = [["function:src/auth.py:validate", 1.0]]

        result = handle_cypher(mock_storage, query)

        mock_storage.execute_raw.assert_called_once_with(query)
        assert "Results (1 rows)" in result
        assert "validate" in result

    def test_rejects_call_install(self, mock_storage):
        """Blocks unsafe CALL INSTALL procedure usage."""
        result = handle_cypher(mock_storage, "CALL INSTALL EXTENSION 'httpfs';")

        assert "Query rejected" in result
        mock_storage.execute_raw.assert_not_called()

    def test_rejects_call_load(self, mock_storage):
        """Blocks unsafe CALL LOAD procedure usage."""
        result = handle_cypher(mock_storage, "CALL LOAD EXTENSION 'json';")

        assert "Query rejected" in result
        mock_storage.execute_raw.assert_not_called()

    def test_rejects_write_keyword_queries(self, mock_storage):
        """Write operations remain blocked."""
        result = handle_cypher(mock_storage, "MATCH (n) DETACH DELETE n RETURN n")

        assert "Query rejected" in result
        mock_storage.execute_raw.assert_not_called()


# ---------------------------------------------------------------------------
# Resource handlers
# ---------------------------------------------------------------------------


class TestResources:
    def test_get_schema(self):
        """Schema resource returns static schema text."""
        from axon.mcp.resources import get_schema

        result = get_schema()
        assert "Node Labels:" in result
        assert "Relationship Types:" in result
        assert "CALLS" in result
        assert "Function" in result

    def test_get_overview(self, mock_storage):
        """Overview resource queries storage for stats."""
        from axon.mcp.resources import get_overview

        mock_storage.execute_raw.return_value = [["Function", 42]]
        result = get_overview(mock_storage)
        assert "Axon Codebase Overview" in result

    def test_get_dead_code_list(self, mock_storage):
        """Dead code resource returns formatted report."""
        from axon.mcp.resources import get_dead_code_list

        mock_storage.execute_raw.return_value = [
            ["old_func", "src/old.py", 10],
        ]
        result = get_dead_code_list(mock_storage)
        assert "Dead Code Report" in result
        assert "old_func" in result

    def test_get_dead_code_list_empty(self, mock_storage):
        """Dead code resource returns clean message when empty."""
        from axon.mcp.resources import get_dead_code_list

        result = get_dead_code_list(mock_storage)
        assert "No dead code detected" in result


# ---------------------------------------------------------------------------
# Confidence tags
# ---------------------------------------------------------------------------


class TestConfidenceTag:
    """_confidence_tag() returns the correct visual indicator."""

    def test_high_confidence(self):
        assert _confidence_tag(1.0) == ""
        assert _confidence_tag(0.95) == ""
        assert _confidence_tag(0.9) == ""

    def test_medium_confidence(self):
        assert _confidence_tag(0.89) == " (~)"
        assert _confidence_tag(0.5) == " (~)"
        assert _confidence_tag(0.7) == " (~)"

    def test_low_confidence(self):
        assert _confidence_tag(0.49) == " (?)"
        assert _confidence_tag(0.1) == " (?)"
        assert _confidence_tag(0.0) == " (?)"


class TestConfidenceInContext:
    """handle_context() displays confidence tags in output."""

    def test_medium_confidence_tag_shown(self, mock_storage_with_relations):
        """Callees with confidence 0.8 show the (~) tag."""
        result = handle_context(mock_storage_with_relations, "validate")
        # _callee has confidence 0.8, which produces " (~)"
        assert "(~)" in result

    def test_high_confidence_no_tag(self, mock_storage_with_relations):
        """Callers with confidence 1.0 show no extra tag."""
        result = handle_context(mock_storage_with_relations, "validate")
        # login_handler has confidence 1.0 — no tag after its line
        assert "login_handler" in result
        # There should be no "(?)" for the high-confidence caller
        lines = result.split("\n")
        caller_line = [l for l in lines if "login_handler" in l][0]
        assert "(?)" not in caller_line
        assert "(~)" not in caller_line


# ---------------------------------------------------------------------------
# Process-grouped search
# ---------------------------------------------------------------------------


class TestGroupByProcess:
    """_group_by_process() groups search results by process membership."""

    def test_empty_results(self, mock_storage):
        """Returns empty dict for empty results list."""
        groups = _group_by_process([], mock_storage)
        assert groups == {}

    def test_with_memberships(self, mock_storage):
        """Returns correct grouping when process memberships exist."""
        results = [
            SearchResult(node_id="func:a", score=1.0, node_name="a"),
            SearchResult(node_id="func:b", score=0.9, node_name="b"),
            SearchResult(node_id="func:c", score=0.8, node_name="c"),
        ]
        mock_storage.get_process_memberships.return_value = {
            "func:a": "Auth Flow",
            "func:c": "Auth Flow",
        }
        groups = _group_by_process(results, mock_storage)
        assert "Auth Flow" in groups
        assert len(groups["Auth Flow"]) == 2

    def test_backend_missing_method(self, mock_storage):
        """Returns empty dict if backend raises AttributeError."""
        mock_storage.get_process_memberships.side_effect = AttributeError
        results = [SearchResult(node_id="func:a", score=1.0)]
        groups = _group_by_process(results, mock_storage)
        assert groups == {}


class TestFormatQueryResults:
    """_format_query_results() renders grouped and ungrouped results."""

    def test_ungrouped_only(self):
        """With no groups, results appear inline."""
        results = [
            SearchResult(
                node_id="func:a", score=1.0, node_name="foo",
                file_path="src/a.py", label="function",
            ),
        ]
        output = _format_query_results(results, {})
        assert "foo (Function)" in output
        assert "src/a.py" in output
        assert "Next:" in output

    def test_with_groups(self):
        """Grouped results appear under process section headers."""
        r1 = SearchResult(
            node_id="func:a", score=1.0, node_name="login",
            file_path="src/auth.py", label="function",
        )
        r2 = SearchResult(
            node_id="func:b", score=0.9, node_name="helper",
            file_path="src/utils.py", label="function",
        )
        groups = {"Auth Flow": [r1]}
        output = _format_query_results([r1, r2], groups)
        assert "=== Auth Flow ===" in output
        assert "=== Other results ===" in output
        assert "login" in output
        assert "helper" in output

    def test_snippet_truncation(self):
        """Snippets longer than 200 chars are truncated."""
        long_snippet = "x" * 300
        results = [
            SearchResult(
                node_id="func:a", score=1.0, node_name="foo",
                file_path="src/a.py", label="function", snippet=long_snippet,
            ),
        ]
        output = _format_query_results(results, {})
        # Snippet in output should be at most 200 chars
        lines = output.split("\n")
        snippet_lines = [l for l in lines if l.strip().startswith("xxx")]
        for line in snippet_lines:
            assert len(line.strip()) <= 200


# ---------------------------------------------------------------------------
# Impact depth grouping
# ---------------------------------------------------------------------------


class TestImpactDepthGrouping:
    """handle_impact() groups results by depth with labels."""

    def test_depth_section_headers(self, mock_storage):
        """Output contains depth section headers with labels."""
        _login = GraphNode(
            id="function:src/api.py:login",
            label=NodeLabel.FUNCTION,
            name="login",
            file_path="src/api.py",
            start_line=5,
            end_line=20,
        )
        _register = GraphNode(
            id="function:src/api.py:register",
            label=NodeLabel.FUNCTION,
            name="register",
            file_path="src/api.py",
            start_line=25,
            end_line=50,
        )
        mock_storage.traverse_with_depth.return_value = [
            (_login, 1), (_register, 2),
        ]
        mock_storage.get_callers_with_confidence.return_value = [(_login, 0.8)]

        result = handle_impact(mock_storage, "validate", depth=2)
        assert "Depth 1" in result
        assert "Direct callers (will break)" in result
        assert "Depth 2" in result
        assert "Indirect (may break)" in result

    def test_depth_3_transitive_label(self, mock_storage):
        """Depth >= 3 shows 'Transitive (review)' label."""
        _node = GraphNode(
            id="function:src/far.py:distant",
            label=NodeLabel.FUNCTION,
            name="distant",
            file_path="src/far.py",
            start_line=1,
            end_line=10,
        )
        mock_storage.traverse_with_depth.return_value = [(_node, 3)]
        mock_storage.get_callers_with_confidence.return_value = []

        result = handle_impact(mock_storage, "validate", depth=3)
        assert "Transitive (review)" in result

    def test_confidence_shown_for_direct_callers(self, mock_storage):
        """Direct callers show inline confidence score."""
        _login = GraphNode(
            id="function:src/api.py:login",
            label=NodeLabel.FUNCTION,
            name="login",
            file_path="src/api.py",
            start_line=5,
            end_line=20,
        )
        mock_storage.traverse_with_depth.return_value = [(_login, 1)]
        mock_storage.get_callers_with_confidence.return_value = [(_login, 0.75)]

        result = handle_impact(mock_storage, "validate", depth=1)
        assert "confidence: 0.75" in result

    def test_depth_clamped_to_max(self, mock_storage):
        """Depth > MAX_TRAVERSE_DEPTH is clamped (no crash)."""
        mock_storage.traverse_with_depth.return_value = []
        result = handle_impact(mock_storage, "validate", depth=100)
        assert "No upstream callers found" in result
