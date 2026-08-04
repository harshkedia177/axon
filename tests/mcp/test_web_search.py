"""Tests for the web search MCP tool."""

from unittest.mock import patch, MagicMock
import json

from axon.mcp.tools import handle_web_search


class TestWebSearch:
    def test_missing_api_key(self) -> None:
        """Test behavior when API key is not set."""
        with patch("os.getenv", return_value=None):
            result = handle_web_search("test query")
            assert "Web search unavailable" in result
            assert "YDC_API_KEY" in result

    def test_empty_query(self) -> None:
        """Test behavior with empty query."""
        with patch("os.getenv", return_value="test-key"):
            result = handle_web_search("")
            assert "Error: Empty search query provided" in result

    def test_invalid_limit(self) -> None:
        """Test that invalid limits are clamped."""
        with patch("os.getenv", return_value="test-key"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.read.return_value = json.dumps({
                    "results": [
                        {
                            "title": "Test Result",
                            "url": "https://example.com",
                            "snippet": "Test snippet"
                        }
                    ]
                }).encode()
                mock_response.__enter__ = lambda x: mock_response
                mock_response.__exit__ = lambda *args: None
                mock_urlopen.return_value = mock_response

                # Test limit too high
                result = handle_web_search("test", limit=100)
                assert "Test Result" in result

                # Test limit too low
                result = handle_web_search("test", limit=-1)
                assert "Test Result" in result

    def test_successful_search(self) -> None:
        """Test successful search with mock response."""
        with patch("os.getenv", return_value="test-key"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.read.return_value = json.dumps({
                    "results": [
                        {
                            "title": "Python Documentation",
                            "url": "https://docs.python.org",
                            "snippet": "Official Python documentation"
                        },
                        {
                            "title": "Python Tutorial",
                            "url": "https://python.org/tutorial", 
                            "snippet": "Learn Python programming"
                        }
                    ]
                }).encode()
                mock_response.__enter__ = lambda x: mock_response
                mock_response.__exit__ = lambda *args: None
                mock_urlopen.return_value = mock_response

                result = handle_web_search("Python docs", limit=2)
                
                assert "Web search results for: Python docs" in result
                assert "Python Documentation" in result
                assert "https://docs.python.org" in result
                assert "Official Python documentation" in result
                assert "Python Tutorial" in result
                assert "external data" in result

    def test_http_error_handling(self) -> None:
        """Test handling of HTTP errors."""
        from urllib.error import HTTPError
        
        with patch("os.getenv", return_value="test-key"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = HTTPError(
                    url="", code=401, msg="Unauthorized", hdrs=None, fp=None
                )
                
                result = handle_web_search("test")
                assert "Invalid API key" in result

    def test_network_error_handling(self) -> None:
        """Test handling of network errors."""
        from urllib.error import URLError
        
        with patch("os.getenv", return_value="test-key"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = URLError("Network unreachable")
                
                result = handle_web_search("test")
                assert "Network connection failed" in result

    def test_no_results(self) -> None:
        """Test behavior when no results are returned."""
        with patch("os.getenv", return_value="test-key"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.read.return_value = json.dumps({"results": []}).encode()
                mock_response.__enter__ = lambda x: mock_response
                mock_response.__exit__ = lambda *args: None
                mock_urlopen.return_value = mock_response

                result = handle_web_search("nonexistent query")
                assert "No search results found" in result

    def test_snippet_truncation(self) -> None:
        """Test that long snippets are truncated properly."""
        with patch("os.getenv", return_value="test-key"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                long_snippet = "A " * 100  # 200 chars
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.read.return_value = json.dumps({
                    "results": [
                        {
                            "title": "Long Result",
                            "url": "https://example.com",
                            "snippet": long_snippet
                        }
                    ]
                }).encode()
                mock_response.__enter__ = lambda x: mock_response
                mock_response.__exit__ = lambda *args: None
                mock_urlopen.return_value = mock_response

                result = handle_web_search("test")
                assert "..." in result  # Truncation marker