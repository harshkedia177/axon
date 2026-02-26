"""Base parser interface and data structure unit tests.

Covers SymbolInfo, ImportInfo, CallInfo, TypeRef, ParseResult dataclasses,
LanguageParser ABC contract, and the get_parser / parse_file functions
from parser_phase.py.
"""

from __future__ import annotations

import pytest

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)
from axon.core.ingestion.parser_phase import get_parser, parse_file, _PARSER_CACHE


# ---------------------------------------------------------------------------
# SymbolInfo
# ---------------------------------------------------------------------------


class TestSymbolInfo:
    """SymbolInfo dataclass construction and defaults."""

    def test_required_fields(self) -> None:
        sym = SymbolInfo(name="foo", kind="function", start_line=1, end_line=10, content="def foo(): pass")
        assert sym.name == "foo"
        assert sym.kind == "function"
        assert sym.start_line == 1
        assert sym.end_line == 10
        assert sym.content == "def foo(): pass"

    def test_defaults(self) -> None:
        sym = SymbolInfo(name="bar", kind="class", start_line=1, end_line=5, content="")
        assert sym.signature == ""
        assert sym.class_name == ""
        assert sym.namespace == ""
        assert sym.decorators == []

    def test_method_with_class(self) -> None:
        sym = SymbolInfo(
            name="do_work",
            kind="method",
            start_line=5,
            end_line=15,
            content="def do_work(self): ...",
            class_name="MyClass",
            decorators=["staticmethod"],
        )
        assert sym.class_name == "MyClass"
        assert sym.decorators == ["staticmethod"]

    def test_php_namespace(self) -> None:
        sym = SymbolInfo(
            name="UserService",
            kind="class",
            start_line=10,
            end_line=50,
            content="class UserService { ... }",
            namespace="App\\Services",
        )
        assert sym.namespace == "App\\Services"


# ---------------------------------------------------------------------------
# ImportInfo
# ---------------------------------------------------------------------------


class TestImportInfo:
    """ImportInfo dataclass construction and defaults."""

    def test_relative_import(self) -> None:
        imp = ImportInfo(module=".utils", names=["helper"], is_relative=True)
        assert imp.module == ".utils"
        assert imp.names == ["helper"]
        assert imp.is_relative is True
        assert imp.alias == ""

    def test_absolute_import_with_alias(self) -> None:
        imp = ImportInfo(module="numpy", names=["array"], is_relative=False, alias="np")
        assert imp.module == "numpy"
        assert imp.alias == "np"
        assert imp.is_relative is False

    def test_defaults(self) -> None:
        imp = ImportInfo(module="os")
        assert imp.names == []
        assert imp.is_relative is False
        assert imp.alias == ""

    def test_php_namespace_import(self) -> None:
        imp = ImportInfo(module="App\\Models\\User", names=["User"], is_relative=False)
        assert imp.module == "App\\Models\\User"


# ---------------------------------------------------------------------------
# CallInfo
# ---------------------------------------------------------------------------


class TestCallInfo:
    """CallInfo dataclass construction and defaults."""

    def test_simple_call(self) -> None:
        call = CallInfo(name="print", line=5)
        assert call.name == "print"
        assert call.line == 5
        assert call.receiver == ""
        assert call.arguments == []

    def test_method_call_with_receiver(self) -> None:
        call = CallInfo(name="save", line=10, receiver="user")
        assert call.receiver == "user"

    def test_call_with_callback_args(self) -> None:
        call = CallInfo(name="map", line=3, arguments=["transform"])
        assert call.arguments == ["transform"]


# ---------------------------------------------------------------------------
# TypeRef
# ---------------------------------------------------------------------------


class TestTypeRef:
    """TypeRef dataclass construction and defaults."""

    def test_param_type(self) -> None:
        ref = TypeRef(name="str", kind="param", line=5, param_name="name")
        assert ref.name == "str"
        assert ref.kind == "param"
        assert ref.param_name == "name"

    def test_return_type(self) -> None:
        ref = TypeRef(name="User", kind="return", line=10)
        assert ref.param_name == ""

    def test_variable_type(self) -> None:
        ref = TypeRef(name="int", kind="variable", line=3)
        assert ref.kind == "variable"


# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------


class TestParseResult:
    """ParseResult dataclass construction and aggregation."""

    def test_empty_construction(self) -> None:
        result = ParseResult()
        assert result.symbols == []
        assert result.imports == []
        assert result.calls == []
        assert result.type_refs == []
        assert result.heritage == []
        assert result.exports == []
        assert result.variable_types == {}

    def test_full_construction(self) -> None:
        result = ParseResult(
            symbols=[SymbolInfo(name="foo", kind="function", start_line=1, end_line=5, content="")],
            imports=[ImportInfo(module="os", names=["path"])],
            calls=[CallInfo(name="print", line=3)],
            type_refs=[TypeRef(name="str", kind="return", line=5)],
            heritage=[("Child", "extends", "Parent")],
            exports=["foo"],
            variable_types={"obj": ["MyClass"]},
        )
        assert len(result.symbols) == 1
        assert len(result.imports) == 1
        assert len(result.calls) == 1
        assert len(result.type_refs) == 1
        assert result.heritage == [("Child", "extends", "Parent")]
        assert result.exports == ["foo"]
        assert result.variable_types == {"obj": ["MyClass"]}


# ---------------------------------------------------------------------------
# LanguageParser ABC
# ---------------------------------------------------------------------------


class TestLanguageParserABC:
    """LanguageParser cannot be instantiated without implementing parse()."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            LanguageParser()  # type: ignore[abstract]

    def test_subclass_must_implement_parse(self) -> None:
        class BadParser(LanguageParser):
            pass

        with pytest.raises(TypeError):
            BadParser()  # type: ignore[abstract]

    def test_subclass_with_parse_works(self) -> None:
        class GoodParser(LanguageParser):
            def parse(self, content: str, file_path: str) -> ParseResult:
                return ParseResult()

        parser = GoodParser()
        result = parser.parse("", "test.py")
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# get_parser (from parser_phase.py)
# ---------------------------------------------------------------------------


class TestGetParser:
    """get_parser returns correct parser instances and caches them."""

    def test_python_parser(self) -> None:
        parser = get_parser("python")
        assert parser is not None
        from axon.core.parsers.python_lang import PythonParser
        assert isinstance(parser, PythonParser)

    def test_typescript_parser(self) -> None:
        parser = get_parser("typescript")
        assert parser is not None

    def test_javascript_parser(self) -> None:
        parser = get_parser("javascript")
        assert parser is not None

    def test_php_parser(self) -> None:
        parser = get_parser("php")
        assert parser is not None
        from axon.core.parsers.php import PhpParser
        assert isinstance(parser, PhpParser)

    def test_html_parser(self) -> None:
        parser = get_parser("html")
        assert parser is not None
        from axon.core.parsers.html import HtmlParser
        assert isinstance(parser, HtmlParser)

    def test_tsx_parser(self) -> None:
        parser = get_parser("tsx")
        assert parser is not None

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language 'ruby'"):
            get_parser("ruby")

    def test_caching(self) -> None:
        p1 = get_parser("python")
        p2 = get_parser("python")
        assert p1 is p2


# ---------------------------------------------------------------------------
# parse_file (from parser_phase.py)
# ---------------------------------------------------------------------------


class TestParseFile:
    """parse_file returns FileParseData with graceful error handling."""

    def test_parse_valid_python(self) -> None:
        fpd = parse_file("src/app.py", "def hello(): pass", "python")
        assert fpd.file_path == "src/app.py"
        assert fpd.language == "python"
        funcs = [s for s in fpd.parse_result.symbols if s.kind == "function"]
        assert any(f.name == "hello" for f in funcs)

    def test_parse_valid_php(self) -> None:
        fpd = parse_file("src/app.php", "<?php\nfunction greet() {}\n", "php")
        assert fpd.file_path == "src/app.php"
        funcs = [s for s in fpd.parse_result.symbols if s.kind == "function"]
        assert any(f.name == "greet" for f in funcs)

    def test_parse_invalid_language_returns_empty(self) -> None:
        fpd = parse_file("src/app.rb", "puts 'hello'", "ruby")
        assert fpd.parse_result.symbols == []
        assert fpd.parse_result.imports == []

    def test_parse_empty_content(self) -> None:
        fpd = parse_file("src/empty.py", "", "python")
        assert fpd.parse_result.symbols == []


# ---------------------------------------------------------------------------
# Language config sync
# ---------------------------------------------------------------------------


class TestLanguageConfigSync:
    """Verify that config/languages.py and parser_phase.get_parser are in sync."""

    def test_all_supported_languages_have_parsers(self) -> None:
        """Every language in SUPPORTED_EXTENSIONS can be parsed."""
        from axon.config.languages import SUPPORTED_EXTENSIONS

        unique_languages = set(SUPPORTED_EXTENSIONS.values())
        for lang in unique_languages:
            # Should not raise ValueError
            parser = get_parser(lang)
            assert parser is not None, f"No parser for language: {lang}"
