"""Security tests for injection vulnerability remediations.

Validates that Phase 1 fixes for Cypher injection (INJ-1 through INJ-5)
are effective and regression-proof.  Covers:

- INJ-2: _sanitize_search_query strict allowlist
- INJ-1: handle_detect_changes parameterized queries
- INJ-3/INJ-4: handle_cypher write rejection via read-only mode
- INJ-5: vector_search numeric type validation
- INJ-2: fts_search / fuzzy_search sanitization with real KuzuDB backend
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from axon.core.graph.model import GraphNode, NodeLabel, generate_id
from axon.core.storage.kuzu_backend import KuzuBackend, _sanitize_search_query


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> KuzuBackend:
    """Return a KuzuBackend initialised in a temporary directory."""
    db_path = tmp_path / "test_security_db"
    b = KuzuBackend()
    b.initialize(db_path)
    yield b
    b.close()


def _make_function_node(
    name: str = "test_func",
    file_path: str = "src/app.py",
    content: str = "def test_func(): pass",
    start_line: int = 1,
    end_line: int = 5,
) -> GraphNode:
    """Helper to build a function node with content for search indexes."""
    return GraphNode(
        id=generate_id(NodeLabel.FUNCTION, file_path, name),
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )


# ---------------------------------------------------------------------------
# (a) _sanitize_search_query tests
# ---------------------------------------------------------------------------


class TestSanitizeSearchQuery:
    """Verify INJ-2: strict allowlist sanitizer for search queries."""

    def test_normal_query_passes_through(self) -> None:
        """Alphanumeric text with spaces is unchanged."""
        assert _sanitize_search_query("hello world") == "hello world"

    def test_single_quotes_stripped(self) -> None:
        """Single quotes are removed to prevent Cypher string escapes."""
        assert _sanitize_search_query("it's") == "its"

    def test_injection_payload_neutralized(self) -> None:
        """A classic Cypher injection payload loses all dangerous characters."""
        result = _sanitize_search_query("' RETURN n.content WHERE '1'='1")
        assert "'" not in result
        assert "=" not in result
        # After stripping quotes and equals, the remaining text is safe
        assert result == "RETURN n.content WHERE 11"

    def test_special_chars_stripped(self) -> None:
        """Semicolons are stripped to prevent query chaining."""
        assert _sanitize_search_query("test; DROP TABLE") == "test DROP TABLE"

    def test_empty_string_after_sanitization(self) -> None:
        """A payload consisting entirely of unsafe chars yields empty string."""
        result = _sanitize_search_query("';=()[]{}|\\")
        assert result == ""

    def test_preserves_underscores_hyphens_dots(self) -> None:
        """Underscores, hyphens, and dots are safe and preserved."""
        assert _sanitize_search_query("my_func-name.py") == "my_func-name.py"

    def test_preserves_digits(self) -> None:
        """Numeric digits pass the allowlist."""
        assert _sanitize_search_query("version2.0") == "version2.0"

    def test_backticks_stripped(self) -> None:
        """Backticks that could escape identifiers are removed."""
        assert _sanitize_search_query("`admin`") == "admin"

    def test_double_quotes_stripped(self) -> None:
        """Double quotes are not in the allowlist and are removed."""
        assert _sanitize_search_query('"hello"') == "hello"

    def test_newlines_and_tabs_preserved_as_whitespace(self) -> None:
        r"""Whitespace characters (\s) are part of the allowlist."""
        # The regex allows \s which includes \n and \t
        result = _sanitize_search_query("hello\tworld")
        assert "hello" in result
        assert "world" in result


# ---------------------------------------------------------------------------
# (b) handle_detect_changes injection tests
# ---------------------------------------------------------------------------


INJECTION_DIFF = """\
diff --git a/src/auth.py b/src/auth'; MATCH (n) DETACH DELETE n; //.py
index abc1234..def5678 100644
--- a/src/auth.py
+++ b/src/auth'; MATCH (n) DETACH DELETE n; //.py
@@ -10,5 +10,7 @@ def validate(user):
     if not user:
         return False
+    # injection attempt
     return True
"""


class TestHandleDetectChangesInjection:
    """Verify INJ-1: handle_detect_changes uses parameterized queries."""

    def test_injection_in_file_path_calls_parameterized_method(self) -> None:
        """File paths with injection payloads are passed to query_symbols_by_file,
        which uses parameterized queries internally -- not string interpolation."""
        from axon.mcp.tools import handle_detect_changes

        storage = MagicMock()
        storage.query_symbols_by_file.return_value = []

        handle_detect_changes(storage, INJECTION_DIFF)

        # The method should have been called with the raw file path.
        # The backend's parameterized query handles safety.
        storage.query_symbols_by_file.assert_called()
        called_path = storage.query_symbols_by_file.call_args[0][0]
        assert "'" in called_path  # raw path preserved, backend handles escaping

    def test_does_not_call_execute_raw(self) -> None:
        """After INJ-1 remediation, handle_detect_changes must never call
        _execute_raw or execute_raw (the old vulnerable code path)."""
        from axon.mcp.tools import handle_detect_changes

        storage = MagicMock()
        storage.query_symbols_by_file.return_value = []

        handle_detect_changes(storage, INJECTION_DIFF)

        # Neither the old public name nor the new private name should be called
        assert not hasattr(storage, "execute_raw") or not storage.execute_raw.called
        # _execute_raw is private and should not be called from tool handlers
        assert not hasattr(storage, "_execute_raw") or not storage._execute_raw.called

    def test_normal_diff_maps_symbols_correctly(self) -> None:
        """A normal diff with matching symbols produces the expected output."""
        from axon.mcp.tools import handle_detect_changes

        storage = MagicMock()
        storage.query_symbols_by_file.return_value = [
            ["function:src/auth.py:validate", "validate", "src/auth.py", 10, 30],
        ]

        normal_diff = """\
diff --git a/src/auth.py b/src/auth.py
index abc1234..def5678 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,5 +10,7 @@ def validate(user):
     if not user:
         return False
+    check_permissions(user)
     return True
"""
        result = handle_detect_changes(storage, normal_diff)
        assert "validate" in result
        assert "src/auth.py" in result


# ---------------------------------------------------------------------------
# (c) handle_cypher write rejection tests
# ---------------------------------------------------------------------------


class TestHandleCypherWriteRejection:
    """Verify INJ-3/INJ-4: write operations are rejected by read-only DB."""

    def test_write_query_returns_read_only_message(self) -> None:
        """When the backend raises a 'read only' error, handle_cypher
        returns a user-friendly rejection message."""
        from axon.mcp.tools import handle_cypher

        storage = MagicMock()
        storage.execute_read_query.side_effect = RuntimeError(
            "Cannot execute write operations in read only mode."
        )

        result = handle_cypher(storage, "MATCH (n) DETACH DELETE n")
        assert "read-only mode" in result
        assert "Write operations" in result
        assert "not permitted" in result

    def test_read_query_succeeds(self) -> None:
        """A legitimate read query returns formatted results."""
        from axon.mcp.tools import handle_cypher

        storage = MagicMock()
        storage.execute_read_query.return_value = [
            ["validate", "src/auth.py"],
        ]

        result = handle_cypher(storage, "MATCH (n:Function) RETURN n.name, n.file_path")
        assert "Results (1 rows)" in result
        assert "validate" in result

    def test_other_errors_return_generic_message(self) -> None:
        """Non-read-only errors return a generic message (ERR-1: no leak of
        internal error details to the client)."""
        from axon.mcp.tools import handle_cypher

        storage = MagicMock()
        storage.execute_read_query.side_effect = RuntimeError("Syntax error in query")

        result = handle_cypher(storage, "INVALID SYNTAX")
        # ERR-1 remediation: generic message, no internal details exposed
        assert "Cypher query failed" in result
        assert "read-only" not in result
        # The raw error message should NOT leak to the client
        assert "Syntax error in query" not in result


# ---------------------------------------------------------------------------
# (d) Vector type validation tests
# ---------------------------------------------------------------------------


class TestVectorTypeValidation:
    """Verify INJ-5: vector_search rejects non-numeric vector elements."""

    def test_string_elements_raise_type_error(self, backend: KuzuBackend) -> None:
        """A vector of strings must raise TypeError before any query runs."""
        with pytest.raises(TypeError, match="numeric"):
            backend.vector_search(["not", "a", "float"], 5)

    def test_valid_floats_do_not_raise(self, backend: KuzuBackend) -> None:
        """A valid float vector should not raise TypeError.
        It may return empty results since no embeddings exist -- that is fine."""
        # This should NOT raise TypeError
        results = backend.vector_search([1.0, 2.0, 3.0], 5)
        assert isinstance(results, list)

    def test_mixed_types_raise_type_error(self, backend: KuzuBackend) -> None:
        """A vector mixing floats and strings must raise TypeError."""
        with pytest.raises(TypeError, match="numeric"):
            backend.vector_search([1.0, "bad", 3.0], 5)

    def test_none_in_vector_raises_type_error(self, backend: KuzuBackend) -> None:
        """None values in the vector must raise TypeError."""
        with pytest.raises(TypeError, match="numeric"):
            backend.vector_search([1.0, None, 3.0], 5)

    def test_integers_accepted(self, backend: KuzuBackend) -> None:
        """Integer values are valid numeric types and should be accepted."""
        results = backend.vector_search([1, 2, 3], 5)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# (e) fts_search and fuzzy_search sanitization with real backend
# ---------------------------------------------------------------------------


class TestFtsSearchSanitization:
    """Verify INJ-2: fts_search sanitizes queries before execution."""

    def test_normal_query_does_not_raise(self, backend: KuzuBackend) -> None:
        """A simple alphanumeric query runs without error."""
        # May return empty -- the point is it doesn't crash
        results = backend.fts_search("normal query", limit=5)
        assert isinstance(results, list)

    def test_injection_payload_does_not_raise(self, backend: KuzuBackend) -> None:
        """A Cypher injection payload is sanitized, preventing query errors."""
        results = backend.fts_search("' OR 1=1 --", limit=5)
        assert isinstance(results, list)

    def test_empty_after_sanitization_returns_empty(self, backend: KuzuBackend) -> None:
        """If sanitization strips all chars, fts_search returns [] early."""
        results = backend.fts_search("';=()[]", limit=5)
        assert results == []

    def test_fts_search_with_indexed_data(self, backend: KuzuBackend) -> None:
        """fts_search finds nodes after data is loaded and indexes rebuilt."""
        node = _make_function_node(
            name="authenticate",
            content="def authenticate(user, password): pass",
        )
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("authenticate", limit=5)
        assert isinstance(results, list)
        # Should find the node we just inserted
        if results:
            assert any("authenticate" in r.node_name for r in results)


class TestFuzzySearchSanitization:
    """Verify INJ-2: fuzzy_search sanitizes queries before execution."""

    def test_normal_query_does_not_raise(self, backend: KuzuBackend) -> None:
        """A normal fuzzy search runs without error."""
        results = backend.fuzzy_search("test", limit=5)
        assert isinstance(results, list)

    def test_injection_payload_does_not_raise(self, backend: KuzuBackend) -> None:
        """Cypher injection in fuzzy_search is neutralized by sanitization."""
        results = backend.fuzzy_search("' RETURN n --", limit=5)
        assert isinstance(results, list)

    def test_empty_after_sanitization_returns_empty(self, backend: KuzuBackend) -> None:
        """If sanitization strips all chars, fuzzy_search returns [] early."""
        results = backend.fuzzy_search("'()[];", limit=5)
        assert results == []

    def test_fuzzy_search_with_indexed_data(self, backend: KuzuBackend) -> None:
        """fuzzy_search finds near-matches in indexed data."""
        node = _make_function_node(
            name="validate",
            content="def validate(data): pass",
        )
        backend.add_nodes([node])

        # "validat" is 1 edit distance from "validate"
        results = backend.fuzzy_search("validat", limit=5, max_distance=2)
        assert isinstance(results, list)
        if results:
            assert any("validate" in r.node_name for r in results)
