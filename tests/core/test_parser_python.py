"""Tests for the Python language parser."""

from __future__ import annotations

import pytest

from axon.core.parsers.python_lang import PythonParser


@pytest.fixture
def parser() -> PythonParser:
    return PythonParser()


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class TestParseSimpleFunction:
    """Parse a standalone function with type annotations."""

    CODE = 'def greet(name: str) -> str:\n    return f"Hello, {name}"\n'

    def test_symbol_count(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        assert len(result.symbols) == 1

    def test_function_name_and_kind(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        func = result.symbols[0]
        assert func.name == "greet"
        assert func.kind == "function"

    def test_function_lines(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        func = result.symbols[0]
        assert func.start_line == 1
        assert func.end_line == 2

    def test_signature_contains_params(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        func = result.symbols[0]
        assert "name: str" in func.signature
        assert "def greet" in func.signature

    def test_signature_contains_return_type(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        func = result.symbols[0]
        assert "-> str" in func.signature

    def test_no_class_name(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        func = result.symbols[0]
        assert func.class_name == ""

    def test_content_includes_body(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        func = result.symbols[0]
        assert "return" in func.content


# ---------------------------------------------------------------------------
# Classes with methods
# ---------------------------------------------------------------------------


class TestParseClassWithMethods:
    """Parse a class that contains methods."""

    CODE = (
        "class User:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n"
        "\n"
        "    def save(self) -> bool:\n"
        "        return True\n"
    )

    def test_symbol_count(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        # 1 class + 2 methods
        assert len(result.symbols) == 3

    def test_class_symbol(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        cls = [s for s in result.symbols if s.kind == "class"]
        assert len(cls) == 1
        assert cls[0].name == "User"

    def test_method_count(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 2

    def test_method_class_name(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        methods = [s for s in result.symbols if s.kind == "method"]
        for method in methods:
            assert method.class_name == "User"

    def test_method_names(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        method_names = {s.name for s in result.symbols if s.kind == "method"}
        assert method_names == {"__init__", "save"}


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestParseInheritance:
    """Parse class inheritance (heritage)."""

    def test_single_parent(self, parser: PythonParser) -> None:
        code = "class Admin(User):\n    pass\n"
        result = parser.parse(code, "test.py")
        assert ("Admin", "extends", "User") in result.heritage

    def test_multiple_parents(self, parser: PythonParser) -> None:
        code = "class Admin(User, Mixin):\n    pass\n"
        result = parser.parse(code, "test.py")
        assert ("Admin", "extends", "User") in result.heritage
        assert ("Admin", "extends", "Mixin") in result.heritage

    def test_no_parent(self, parser: PythonParser) -> None:
        code = "class Plain:\n    pass\n"
        result = parser.parse(code, "test.py")
        assert len(result.heritage) == 0


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestParseImports:
    """Parse import statements."""

    CODE = "import os\nfrom os.path import join\nfrom ..models import User\n"

    def test_import_count(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        assert len(result.imports) == 3

    def test_plain_import(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        os_imp = [i for i in result.imports if i.module == "os"]
        assert len(os_imp) == 1
        assert "os" in os_imp[0].names
        assert os_imp[0].is_relative is False

    def test_from_import(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        path_imp = [i for i in result.imports if i.module == "os.path"]
        assert len(path_imp) == 1
        assert "join" in path_imp[0].names
        assert path_imp[0].is_relative is False

    def test_relative_import(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        rel_imp = [i for i in result.imports if i.is_relative]
        assert len(rel_imp) == 1
        assert "User" in rel_imp[0].names
        assert ".." in rel_imp[0].module

    def test_dotted_import(self, parser: PythonParser) -> None:
        code = "import os.path\n"
        result = parser.parse(code, "test.py")
        assert len(result.imports) == 1
        imp = result.imports[0]
        assert imp.module == "os.path"
        assert "path" in imp.names

    def test_relative_dot_import(self, parser: PythonParser) -> None:
        code = "from . import utils\n"
        result = parser.parse(code, "test.py")
        assert len(result.imports) == 1
        imp = result.imports[0]
        assert imp.module == "."
        assert "utils" in imp.names
        assert imp.is_relative is True

    def test_from_import_multiple_names(self, parser: PythonParser) -> None:
        code = "from os.path import join, exists\n"
        result = parser.parse(code, "test.py")
        assert len(result.imports) == 1
        imp = result.imports[0]
        assert imp.module == "os.path"
        assert "join" in imp.names
        assert "exists" in imp.names


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------


class TestParseFunctionCalls:
    """Parse function and method calls."""

    CODE = 'def process():\n    result = validate(data)\n    user.save()\n    print("done")\n'

    def test_simple_call(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        validate_calls = [c for c in result.calls if c.name == "validate"]
        assert len(validate_calls) == 1
        assert validate_calls[0].receiver == ""

    def test_method_call(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        save_calls = [c for c in result.calls if c.name == "save"]
        assert len(save_calls) == 1
        assert save_calls[0].receiver == "user"

    def test_builtin_call(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        print_calls = [c for c in result.calls if c.name == "print"]
        assert len(print_calls) == 1

    def test_self_method_call(self, parser: PythonParser) -> None:
        code = "class Foo:\n    def run(self):\n        self.validate()\n"
        result = parser.parse(code, "test.py")
        validate_calls = [c for c in result.calls if c.name == "validate"]
        assert len(validate_calls) == 1
        assert validate_calls[0].receiver == "self"

    def test_chained_call(self, parser: PythonParser) -> None:
        code = "obj.method1().method2()\n"
        result = parser.parse(code, "test.py")
        # The outer call is method2, the inner is method1
        m2 = [c for c in result.calls if c.name == "method2"]
        assert len(m2) == 1
        m1 = [c for c in result.calls if c.name == "method1"]
        assert len(m1) == 1
        assert m1[0].receiver == "obj"


# ---------------------------------------------------------------------------
# Type annotations
# ---------------------------------------------------------------------------


class TestParseTypeAnnotations:
    """Parse type annotations from parameters, return types, and variables."""

    CODE = (
        "def handle(user: User, config: Config) -> Response:\n"
        "    result: AuthResult = authenticate(user)\n"
    )

    def test_param_types(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        param_refs = [t for t in result.type_refs if t.kind == "param"]
        param_names = {t.name for t in param_refs}
        assert "User" in param_names
        assert "Config" in param_names

    def test_param_names_attached(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        user_ref = [t for t in result.type_refs if t.name == "User" and t.kind == "param"]
        assert len(user_ref) == 1
        assert user_ref[0].param_name == "user"

        config_ref = [t for t in result.type_refs if t.name == "Config" and t.kind == "param"]
        assert len(config_ref) == 1
        assert config_ref[0].param_name == "config"

    def test_return_type(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        return_refs = [t for t in result.type_refs if t.kind == "return"]
        assert any(t.name == "Response" for t in return_refs)

    def test_variable_annotation(self, parser: PythonParser) -> None:
        result = parser.parse(self.CODE, "test.py")
        var_refs = [t for t in result.type_refs if t.kind == "variable"]
        assert any(t.name == "AuthResult" for t in var_refs)

    def test_builtin_types_skipped(self, parser: PythonParser) -> None:
        code = "def foo(x: int, y: str) -> bool:\n    pass\n"
        result = parser.parse(code, "test.py")
        # int, str, bool are all built-in — should produce no type_refs.
        assert len(result.type_refs) == 0

    def test_module_level_variable_annotation(self, parser: PythonParser) -> None:
        code = "config: AppConfig = load_config()\n"
        result = parser.parse(code, "test.py")
        var_refs = [t for t in result.type_refs if t.kind == "variable"]
        assert any(t.name == "AppConfig" for t in var_refs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and less common patterns."""

    def test_empty_file(self, parser: PythonParser) -> None:
        result = parser.parse("", "empty.py")
        assert result.symbols == []
        assert result.imports == []
        assert result.calls == []
        assert result.type_refs == []
        assert result.heritage == []

    def test_syntax_error_does_not_crash(self, parser: PythonParser) -> None:
        code = "def broken(\n"
        # Should not raise; tree-sitter produces a partial tree.
        result = parser.parse(code, "broken.py")
        assert isinstance(result, type(result))

    def test_nested_function(self, parser: PythonParser) -> None:
        code = "def outer():\n    def inner():\n        pass\n"
        result = parser.parse(code, "test.py")
        names = {s.name for s in result.symbols}
        assert "outer" in names
        # inner is inside a function body — should not have class_name
        inner = [s for s in result.symbols if s.name == "inner"]
        assert len(inner) == 1
        assert inner[0].class_name == ""
        assert inner[0].kind == "function"

    def test_decorator_does_not_affect_parsing(self, parser: PythonParser) -> None:
        code = "class Service:\n    @staticmethod\n    def create() -> None:\n        pass\n"
        result = parser.parse(code, "test.py")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "create"
        assert methods[0].class_name == "Service"


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


class TestParseDecorators:
    """Decorator names are captured on SymbolInfo."""

    def test_simple_decorator(self, parser: PythonParser) -> None:
        code = "@staticmethod\ndef create() -> None:\n    pass\n"
        result = parser.parse(code, "test.py")
        assert len(result.symbols) == 1
        assert result.symbols[0].decorators == ["staticmethod"]

    def test_dotted_decorator(self, parser: PythonParser) -> None:
        code = "@app.route\ndef index():\n    pass\n"
        result = parser.parse(code, "test.py")
        assert result.symbols[0].decorators == ["app.route"]

    def test_decorator_with_call(self, parser: PythonParser) -> None:
        code = "@server.list_tools()\nasync def list_tools():\n    return []\n"
        result = parser.parse(code, "test.py")
        assert result.symbols[0].decorators == ["server.list_tools"]

    def test_multiple_decorators(self, parser: PythonParser) -> None:
        code = "@staticmethod\n@cache\ndef compute():\n    pass\n"
        result = parser.parse(code, "test.py")
        assert result.symbols[0].decorators == ["staticmethod", "cache"]

    def test_undecorated_function_has_empty_decorators(self, parser: PythonParser) -> None:
        code = "def plain():\n    pass\n"
        result = parser.parse(code, "test.py")
        assert result.symbols[0].decorators == []

    def test_decorated_method_in_class(self, parser: PythonParser) -> None:
        code = "class Service:\n    @staticmethod\n    def create() -> None:\n        pass\n"
        result = parser.parse(code, "test.py")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].decorators == ["staticmethod"]
        # The class itself should NOT have the method's decorators.
        classes = [s for s in result.symbols if s.kind == "class"]
        assert classes[0].decorators == []

    def test_decorated_class(self, parser: PythonParser) -> None:
        code = "@dataclass\nclass Config:\n    name: str\n"
        result = parser.parse(code, "test.py")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].decorators == ["dataclass"]
