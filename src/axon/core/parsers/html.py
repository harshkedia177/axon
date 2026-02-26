"""HTML parser for Axon.

Extracts cross-boundary references between HTML and JavaScript:
- ``<script src="...">`` tags as ImportInfo (file-level dependencies)
- Inline event handler attributes (onclick, onchange, etc.) as CallInfo
- Inline ``<script>`` block contents delegated to the JavaScript parser

Does not use tree-sitter — Python's stdlib ``html.parser`` is sufficient
for the three extraction patterns needed.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
)

_EVENT_HANDLER_ATTRS: frozenset[str] = frozenset({
    "onclick", "ondblclick", "onmousedown", "onmouseup", "onmouseover",
    "onmousemove", "onmouseout", "onkeypress", "onkeydown", "onkeyup",
    "onfocus", "onblur", "onchange", "onsubmit", "onreset", "onselect",
    "onload", "onunload", "onerror", "onresize", "onscroll", "oninput",
    "oncontextmenu", "ondrag", "ondrop", "ontouchstart", "ontouchend",
})

# Regex to extract function call names from event handler attribute values.
# Matches patterns like: ``functionName(`` or ``obj.methodName(``.
_CALL_NAME_RE = re.compile(r"(?:^|[;\s])(\w+)\s*\(")


class _HtmlWalker(HTMLParser):
    """HTMLParser subclass that collects script sources and event handler calls."""

    def __init__(self) -> None:
        super().__init__()
        self.imports: list[ImportInfo] = []
        self.calls: list[CallInfo] = []
        self._in_script = False
        self._script_start_line = 0
        self._script_content_parts: list[str] = []
        self._inline_scripts: list[tuple[int, str]] = []  # (start_line, content)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        line = self.getpos()[0]

        if tag == "script":
            src = None
            for attr_name, attr_val in attrs:
                if attr_name == "src" and attr_val:
                    src = attr_val
                    break

            if src:
                self.imports.append(
                    ImportInfo(
                        module=src,
                        names=[],
                        is_relative=src.startswith(".") or not src.startswith(("http://", "https://", "//")),
                    )
                )
            else:
                # Inline script — collect content for JS parsing
                self._in_script = True
                self._script_start_line = line
                self._script_content_parts = []

        elif tag == "link":
            # Capture <link rel="stylesheet" href="..."> as imports
            rel = None
            href = None
            for attr_name, attr_val in attrs:
                if attr_name == "rel":
                    rel = attr_val
                elif attr_name == "href" and attr_val:
                    href = attr_val
            if rel == "stylesheet" and href:
                self.imports.append(
                    ImportInfo(
                        module=href,
                        names=[],
                        is_relative=href.startswith(".") or not href.startswith(("http://", "https://", "//")),
                    )
                )

        # Extract event handler attributes
        for attr_name, attr_val in attrs:
            if attr_name and attr_name.lower() in _EVENT_HANDLER_ATTRS and attr_val:
                for match in _CALL_NAME_RE.finditer(attr_val):
                    func_name = match.group(1)
                    # Skip common JS noise
                    if func_name not in ("return", "if", "else", "var", "let", "const", "this", "event", "true", "false"):
                        self.calls.append(
                            CallInfo(name=func_name, line=line)
                        )

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script:
            self._in_script = False
            content = "".join(self._script_content_parts)
            if content.strip():
                self._inline_scripts.append((self._script_start_line, content))
            self._script_content_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_content_parts.append(data)


class HtmlParser(LanguageParser):
    """Parse HTML files to extract JS cross-boundary references.

    Finds ``<script src>`` tags, inline event handlers, and inline
    ``<script>`` blocks (delegated to the JavaScript parser).
    """

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse HTML content and return references to JavaScript symbols."""
        walker = _HtmlWalker()
        try:
            walker.feed(content)
        except Exception:
            pass  # Malformed HTML — return whatever we collected

        result = ParseResult()
        result.imports = walker.imports
        result.calls = walker.calls

        # Parse inline <script> blocks with the JS parser
        if walker._inline_scripts:
            try:
                from axon.core.parsers.typescript import TypeScriptParser

                js_parser = TypeScriptParser(dialect="javascript")
                for start_line, script_content in walker._inline_scripts:
                    js_result = js_parser.parse(script_content, file_path)
                    # Offset line numbers to match the HTML file
                    offset = start_line
                    for sym in js_result.symbols:
                        sym.start_line += offset
                        sym.end_line += offset
                        result.symbols.append(sym)
                    for call in js_result.calls:
                        call.line += offset
                        result.calls.append(call)
                    result.imports.extend(js_result.imports)
                    result.exports.extend(js_result.exports)
            except Exception:
                pass  # If JS parsing fails, we still have the HTML-level results

        return result
