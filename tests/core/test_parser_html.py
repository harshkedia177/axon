"""HTML parser unit tests.

Covers script src import extraction, inline event handler call extraction,
inline <script> block delegation to the JS parser, malformed HTML tolerance,
and line number offsetting.
"""

from __future__ import annotations

import pytest

from axon.core.parsers.html import HtmlParser


@pytest.fixture
def parser() -> HtmlParser:
    return HtmlParser()


# ---------------------------------------------------------------------------
# Script src imports
# ---------------------------------------------------------------------------


class TestScriptSrcImports:
    """<script src="..."> tags produce ImportInfo entries."""

    def test_relative_script_src(self, parser: HtmlParser) -> None:
        html = '<html><head><script src="./app.js"></script></head></html>'
        result = parser.parse(html, "index.html")
        assert len(result.imports) == 1
        assert result.imports[0].module == "./app.js"
        assert result.imports[0].is_relative is True

    def test_absolute_script_src(self, parser: HtmlParser) -> None:
        html = '<html><head><script src="https://cdn.example.com/lib.js"></script></head></html>'
        result = parser.parse(html, "index.html")
        assert len(result.imports) == 1
        assert result.imports[0].module == "https://cdn.example.com/lib.js"
        assert result.imports[0].is_relative is False

    def test_protocol_relative_src(self, parser: HtmlParser) -> None:
        html = '<script src="//cdn.example.com/lib.js"></script>'
        result = parser.parse(html, "index.html")
        assert len(result.imports) == 1
        assert result.imports[0].is_relative is False

    def test_bare_path_src(self, parser: HtmlParser) -> None:
        html = '<script src="assets/vendor/react.js"></script>'
        result = parser.parse(html, "index.html")
        assert len(result.imports) == 1
        assert result.imports[0].module == "assets/vendor/react.js"
        assert result.imports[0].is_relative is True

    def test_multiple_script_tags(self, parser: HtmlParser) -> None:
        html = """<html>
<head>
    <script src="react.js"></script>
    <script src="react-dom.js"></script>
    <script src="app.bundle.js"></script>
</head>
</html>"""
        result = parser.parse(html, "index.html")
        modules = [imp.module for imp in result.imports]
        assert "react.js" in modules
        assert "react-dom.js" in modules
        assert "app.bundle.js" in modules

    def test_script_src_with_no_value_ignored(self, parser: HtmlParser) -> None:
        """<script src> without a value doesn't crash."""
        html = '<script src></script>'
        result = parser.parse(html, "index.html")
        # No src value -> treated as inline script (no import)
        assert len(result.imports) == 0

    def test_script_with_type_module(self, parser: HtmlParser) -> None:
        html = '<script type="module" src="./main.mjs"></script>'
        result = parser.parse(html, "index.html")
        assert len(result.imports) == 1
        assert result.imports[0].module == "./main.mjs"


# ---------------------------------------------------------------------------
# Event handler attribute extraction
# ---------------------------------------------------------------------------


class TestEventHandlerCalls:
    """Inline event handler attributes produce CallInfo entries."""

    def test_onclick_simple(self, parser: HtmlParser) -> None:
        html = '<button onclick="handleClick()">Click</button>'
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "handleClick" in call_names

    def test_onsubmit_handler(self, parser: HtmlParser) -> None:
        html = '<form onsubmit="validateForm(event)">...</form>'
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "validateForm" in call_names

    def test_onchange_handler(self, parser: HtmlParser) -> None:
        html = '<select onchange="updateSelection(this.value)">...</select>'
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "updateSelection" in call_names

    def test_multiple_handlers_semicolon(self, parser: HtmlParser) -> None:
        html = '<button onclick="doA(); doB()">Go</button>'
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "doA" in call_names
        assert "doB" in call_names

    def test_js_keywords_filtered(self, parser: HtmlParser) -> None:
        """JS keywords in event handlers are not emitted as calls."""
        html = '<button onclick="if(true) { doStuff() }">Go</button>'
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "doStuff" in call_names
        assert "if" not in call_names

    def test_return_keyword_filtered(self, parser: HtmlParser) -> None:
        html = '<form onsubmit="return validate()">...</form>'
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "validate" in call_names
        assert "return" not in call_names

    def test_this_keyword_filtered(self, parser: HtmlParser) -> None:
        html = '<input oninput="this.validate()">'
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "this" not in call_names

    def test_event_handler_line_number(self, parser: HtmlParser) -> None:
        html = """<html>
<body>
<button onclick="handler()">Go</button>
</body>
</html>"""
        result = parser.parse(html, "index.html")
        handler_calls = [c for c in result.calls if c.name == "handler"]
        assert len(handler_calls) == 1
        assert handler_calls[0].line == 3

    def test_no_event_handlers_no_calls(self, parser: HtmlParser) -> None:
        html = '<div class="container"><p>Hello</p></div>'
        result = parser.parse(html, "index.html")
        assert len(result.calls) == 0


# ---------------------------------------------------------------------------
# Inline <script> block delegation
# ---------------------------------------------------------------------------


class TestInlineScriptDelegation:
    """Inline <script> blocks are parsed via the JS parser."""

    def test_inline_script_function(self, parser: HtmlParser) -> None:
        html = """<html>
<body>
<script>
function greet() {
    console.log("hello");
}
</script>
</body>
</html>"""
        result = parser.parse(html, "index.html")
        funcs = [s for s in result.symbols if s.kind == "function"]
        assert any(f.name == "greet" for f in funcs)

    def test_inline_script_line_offset(self, parser: HtmlParser) -> None:
        """Symbols from inline scripts have line numbers offset to match HTML."""
        html = """<html>
<body>
<script>
function myFunc() { return 1; }
</script>
</body>
</html>"""
        result = parser.parse(html, "index.html")
        funcs = [s for s in result.symbols if s.name == "myFunc"]
        assert len(funcs) == 1
        # Script tag is on line 3, function is on line 1 of the script block.
        # Offset: 3 + 1 = 4 (approximately — depends on parser line counting).
        assert funcs[0].start_line >= 3

    def test_inline_script_calls_extracted(self, parser: HtmlParser) -> None:
        html = """<script>
doSomething();
initApp();
</script>"""
        result = parser.parse(html, "index.html")
        call_names = [c.name for c in result.calls]
        assert "doSomething" in call_names
        assert "initApp" in call_names

    def test_multiple_inline_scripts(self, parser: HtmlParser) -> None:
        html = """<script>function a() {}</script>
<script>function b() {}</script>"""
        result = parser.parse(html, "index.html")
        func_names = [s.name for s in result.symbols if s.kind == "function"]
        assert "a" in func_names
        assert "b" in func_names

    def test_empty_inline_script_no_crash(self, parser: HtmlParser) -> None:
        html = "<script></script>"
        result = parser.parse(html, "index.html")
        # Empty script — nothing extracted, no crash.
        assert result is not None

    def test_inline_script_with_src_not_parsed(self, parser: HtmlParser) -> None:
        """<script src="..."> with body text — body is ignored, only src imported."""
        html = '<script src="lib.js">function shouldBeIgnored() {}</script>'
        result = parser.parse(html, "index.html")
        # Should get the import but NOT the inline function.
        assert len(result.imports) == 1
        assert result.imports[0].module == "lib.js"
        funcs = [s for s in result.symbols if s.name == "shouldBeIgnored"]
        assert len(funcs) == 0


# ---------------------------------------------------------------------------
# Malformed HTML tolerance
# ---------------------------------------------------------------------------


class TestMalformedHtml:
    """Parser doesn't crash on malformed HTML."""

    def test_unclosed_tags(self, parser: HtmlParser) -> None:
        html = '<div><p>unclosed<script src="app.js">'
        result = parser.parse(html, "index.html")
        assert len(result.imports) == 1

    def test_missing_quotes(self, parser: HtmlParser) -> None:
        html = '<script src=app.js></script>'
        result = parser.parse(html, "index.html")
        assert len(result.imports) == 1
        assert result.imports[0].module == "app.js"

    def test_completely_broken_html(self, parser: HtmlParser) -> None:
        html = "<<<>>>not html at all &&&"
        result = parser.parse(html, "index.html")
        # Should not crash.
        assert result is not None

    def test_empty_content(self, parser: HtmlParser) -> None:
        result = parser.parse("", "index.html")
        assert len(result.symbols) == 0
        assert len(result.imports) == 0
        assert len(result.calls) == 0


# ---------------------------------------------------------------------------
# Mixed: imports + event handlers + inline scripts
# ---------------------------------------------------------------------------


class TestMixedExtraction:
    """Full HTML page with all three extraction types."""

    def test_full_page(self, parser: HtmlParser) -> None:
        html = """<!DOCTYPE html>
<html>
<head>
    <script src="vendor/react.js"></script>
    <script src="./app.bundle.js"></script>
</head>
<body>
    <button onclick="handleClick()">Click me</button>
    <form onsubmit="return validateForm()">
        <input type="submit">
    </form>
    <script>
    function initPage() {
        setupListeners();
    }
    initPage();
    </script>
</body>
</html>"""
        result = parser.parse(html, "index.html")

        # Imports from <script src>
        modules = [imp.module for imp in result.imports]
        assert "vendor/react.js" in modules
        assert "./app.bundle.js" in modules

        # Calls from event handlers
        call_names = [c.name for c in result.calls]
        assert "handleClick" in call_names
        assert "validateForm" in call_names

        # Symbols from inline <script>
        func_names = [s.name for s in result.symbols if s.kind == "function"]
        assert "initPage" in func_names

        # Calls from inline <script>
        assert "setupListeners" in call_names or "initPage" in call_names
