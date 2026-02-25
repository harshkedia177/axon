"""PHP parser using tree-sitter.

Extracts symbols (functions, classes, methods, interfaces, enums),
imports (use statements), call expressions, type annotation references,
and heritage (extends / implements) relationships from PHP source files.
"""

from __future__ import annotations

import tree_sitter_php as tsphp
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)

PHP_LANGUAGE = Language(tsphp.language_php())

_BUILTIN_TYPES: frozenset[str] = frozenset({
    "string", "int", "float", "bool", "array", "object", "null",
    "void", "never", "mixed", "callable", "iterable", "self",
    "static", "parent", "true", "false",
})

_MAGIC_METHODS: frozenset[str] = frozenset({
    "__construct", "__destruct", "__call", "__callStatic", "__get",
    "__set", "__isset", "__unset", "__sleep", "__wakeup", "__serialize",
    "__unserialize", "__toString", "__invoke", "__set_state", "__clone",
    "__debugInfo",
})


class PhpParser(LanguageParser):
    """Parse PHP source files via tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(PHP_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse PHP source and return structured information."""
        tree = self._parser.parse(content.encode("utf-8"))
        result = ParseResult()
        self._walk(tree.root_node, content, result, class_name="")
        return result

    def _walk(
        self, node: Node, source: str, result: ParseResult, class_name: str
    ) -> None:
        """Recursively walk the AST to extract definitions, imports, and calls."""
        for child in node.children:
            ntype = child.type

            if ntype == "function_definition":
                self._extract_function(child, source, result)
            elif ntype == "class_declaration":
                self._extract_class(child, source, result)
            elif ntype == "interface_declaration":
                self._extract_interface(child, source, result)
            elif ntype == "enum_declaration":
                self._extract_enum(child, source, result)
            elif ntype == "method_declaration":
                self._extract_method(child, source, result, class_name)
            elif ntype == "namespace_use_declaration":
                self._extract_use(child, result)
            elif ntype == "function_call_expression":
                self._extract_function_call(child, result)
            elif ntype == "member_call_expression":
                self._extract_member_call(child, result)
            elif ntype == "scoped_call_expression":
                self._extract_scoped_call(child, result)
            elif ntype in ("expression_statement", "return_statement",
                           "compound_statement", "if_statement",
                           "else_clause", "else_if_clause",
                           "while_statement", "for_statement",
                           "foreach_statement", "switch_statement",
                           "switch_block", "case_statement",
                           "default_statement",
                           "try_statement", "catch_clause",
                           "finally_clause",
                           "program", "namespace_definition",
                           "declaration_list", "enum_declaration_list",
                           "parenthesized_expression", "assignment_expression",
                           "binary_expression", "unary_op_expression",
                           "match_expression", "match_condition_list",
                           "match_conditional_expression"):
                self._walk(child, source, result, class_name)
            elif ntype == "object_creation_expression":
                self._extract_new_expression(child, result)
            elif ntype in ("require_expression", "require_once_expression",
                           "include_expression", "include_once_expression"):
                self._extract_include(child, result)

    def _extract_function(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Extract a top-level function definition."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()
        signature = self._build_signature(node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="function",
                start_line=start_line,
                end_line=end_line,
                content=content,
                signature=signature,
            )
        )

        self._extract_param_types(node, result)
        self._extract_return_type(node, result)

        # Walk the function body for nested calls
        body = node.child_by_field_name("body")
        if body:
            self._walk(body, source, result, class_name="")

    def _extract_class(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Extract a class declaration with heritage."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        class_name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=class_name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        # Extract extends and implements
        for child in node.children:
            if child.type == "base_clause":
                for sub in child.children:
                    if sub.type == "name":
                        result.heritage.append((class_name, "extends", sub.text.decode()))
                    elif sub.type == "qualified_name":
                        result.heritage.append((class_name, "extends", self._qualified_name(sub)))
            elif child.type == "class_interface_clause":
                for sub in child.children:
                    if sub.type == "name":
                        result.heritage.append((class_name, "implements", sub.text.decode()))
                    elif sub.type == "qualified_name":
                        result.heritage.append((class_name, "implements", self._qualified_name(sub)))
            elif child.type == "declaration_list":
                self._walk(child, source, result, class_name=class_name)

    def _extract_interface(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Extract an interface declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="interface",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        # Extract extends (interfaces can extend other interfaces)
        for child in node.children:
            if child.type == "base_clause":
                for sub in child.children:
                    if sub.type == "name":
                        result.heritage.append((name, "extends", sub.text.decode()))
            elif child.type == "declaration_list":
                self._walk(child, source, result, class_name=name)

    def _extract_enum(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Extract a PHP 8.1+ enum declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="enum",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

    def _extract_method(
        self, node: Node, source: str, result: ParseResult, class_name: str
    ) -> None:
        """Extract a method declaration inside a class or interface."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()
        signature = self._build_signature(node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=start_line,
                end_line=end_line,
                content=content,
                signature=signature,
                class_name=class_name,
            )
        )

        self._extract_param_types(node, result)
        self._extract_return_type(node, result)

        # Walk method body for calls
        body = node.child_by_field_name("body")
        if body:
            self._walk(body, source, result, class_name=class_name)

    def _extract_use(self, node: Node, result: ParseResult) -> None:
        """Extract a ``use`` statement (namespace import)."""
        for child in node.children:
            if child.type == "namespace_use_clause":
                qname_node = None
                alias = ""
                for sub in child.children:
                    if sub.type == "qualified_name":
                        qname_node = sub
                    elif sub.type == "namespace_aliasing_clause":
                        for alias_child in sub.children:
                            if alias_child.type == "name":
                                alias = alias_child.text.decode()

                if qname_node is not None:
                    full_name = self._qualified_name(qname_node)
                    # The imported name is the last segment (e.g., "UserService" from "App\Services\UserService")
                    parts = full_name.replace("\\", "/").split("/")
                    short_name = alias or parts[-1]
                    result.imports.append(
                        ImportInfo(
                            module=full_name,
                            names=[short_name],
                            is_relative=False,
                            alias=alias,
                        )
                    )
            elif child.type == "namespace_use_group":
                # use App\Models\{User, Role};
                prefix = ""
                for sub in child.children:
                    if sub.type == "qualified_name" or sub.type == "namespace_name":
                        prefix = sub.text.decode()
                    elif sub.type == "namespace_use_group_clause":
                        for clause_child in sub.children:
                            if clause_child.type == "namespace_use_clause":
                                for inner in clause_child.children:
                                    if inner.type == "name":
                                        name = inner.text.decode()
                                        full = f"{prefix}\\{name}" if prefix else name
                                        result.imports.append(
                                            ImportInfo(
                                                module=full,
                                                names=[name],
                                                is_relative=False,
                                            )
                                        )

    def _extract_function_call(self, node: Node, result: ParseResult) -> None:
        """Extract a function call expression."""
        func_node = node.child_by_field_name("function")
        if func_node is None:
            return

        line = node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(node)

        if func_node.type == "name":
            result.calls.append(
                CallInfo(name=func_node.text.decode(), line=line, arguments=arguments)
            )
        elif func_node.type == "qualified_name":
            name = self._qualified_name(func_node)
            # Use just the last segment as the call name
            short = name.rsplit("\\", 1)[-1] if "\\" in name else name
            result.calls.append(
                CallInfo(name=short, line=line, arguments=arguments)
            )

    def _extract_member_call(self, node: Node, result: ParseResult) -> None:
        """Extract ``$obj->method()`` calls."""
        name_node = node.child_by_field_name("name")
        obj_node = node.child_by_field_name("object")
        if name_node is None:
            return

        line = node.start_point[0] + 1
        method_name = name_node.text.decode()
        receiver = ""
        if obj_node is not None:
            receiver = obj_node.text.decode().lstrip("$")

        result.calls.append(
            CallInfo(name=method_name, line=line, receiver=receiver)
        )

    def _extract_scoped_call(self, node: Node, result: ParseResult) -> None:
        """Extract ``ClassName::method()`` (static calls)."""
        name_node = node.child_by_field_name("name")
        scope_node = node.child_by_field_name("scope")
        if name_node is None:
            return

        line = node.start_point[0] + 1
        method_name = name_node.text.decode()
        receiver = scope_node.text.decode() if scope_node else ""

        result.calls.append(
            CallInfo(name=method_name, line=line, receiver=receiver)
        )

    def _extract_new_expression(self, node: Node, result: ParseResult) -> None:
        """Extract ``new ClassName(...)`` — emit a call to the class."""
        for child in node.children:
            if child.type == "name":
                result.calls.append(
                    CallInfo(name=child.text.decode(), line=node.start_point[0] + 1)
                )
                break
            elif child.type == "qualified_name":
                name = self._qualified_name(child)
                short = name.rsplit("\\", 1)[-1] if "\\" in name else name
                result.calls.append(
                    CallInfo(name=short, line=node.start_point[0] + 1)
                )
                break

    def _extract_param_types(self, func_node: Node, result: ParseResult) -> None:
        """Extract type hints from function/method parameters."""
        params = func_node.child_by_field_name("parameters")
        if params is None:
            return

        for param in params.children:
            if param.type == "simple_parameter":
                self._extract_param_type(param, result)

    def _extract_param_type(self, param_node: Node, result: ParseResult) -> None:
        """Extract a single parameter's type hint."""
        param_name = ""
        type_name = ""

        for child in param_node.children:
            if child.type == "variable_name":
                # Get the name without $
                for sub in child.children:
                    if sub.type == "name":
                        param_name = sub.text.decode()
            elif child.type in ("named_type", "optional_type"):
                type_name = self._extract_type_name_from_node(child)
            elif child.type == "primitive_type":
                type_name = child.text.decode()

        if type_name and type_name.lower() not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="param",
                    line=param_node.start_point[0] + 1,
                    param_name=param_name,
                )
            )

    def _extract_return_type(self, func_node: Node, result: ParseResult) -> None:
        """Extract return type hint from a function/method."""
        for child in func_node.children:
            if child.type in ("named_type", "optional_type"):
                type_name = self._extract_type_name_from_node(child)
                if type_name and type_name.lower() not in _BUILTIN_TYPES:
                    result.type_refs.append(
                        TypeRef(
                            name=type_name,
                            kind="return",
                            line=child.start_point[0] + 1,
                        )
                    )
            elif child.type == "primitive_type":
                pass  # Skip built-in return types

    @staticmethod
    def _extract_type_name_from_node(type_node: Node) -> str:
        """Extract a type name from a named_type or optional_type node."""
        for child in type_node.children:
            if child.type == "name":
                return child.text.decode()
            elif child.type == "qualified_name":
                text = child.text.decode()
                return text.rsplit("\\", 1)[-1] if "\\" in text else text
            elif child.type == "named_type":
                return PhpParser._extract_type_name_from_node(child)
        return type_node.text.decode()

    @staticmethod
    def _qualified_name(node: Node) -> str:
        """Build a qualified name string from a qualified_name node."""
        parts: list[str] = []
        for child in node.children:
            if child.type == "namespace_name":
                for sub in child.children:
                    if sub.type == "name":
                        parts.append(sub.text.decode())
            elif child.type == "name":
                parts.append(child.text.decode())
        return "\\".join(parts)

    @staticmethod
    def _build_signature(node: Node, name: str) -> str:
        """Build a human-readable signature for a function/method."""
        params_node = node.child_by_field_name("parameters")
        params_text = params_node.text.decode() if params_node else "()"

        return_type = ""
        for child in node.children:
            if child.type in ("named_type", "optional_type", "primitive_type"):
                # Check it comes after the parameters (return type position)
                if params_node and child.start_byte > params_node.end_byte:
                    return_type = child.text.decode()
                    break

        sig = f"function {name}{params_text}"
        if return_type:
            sig += f": {return_type}"
        return sig

    def _extract_include(self, node: Node, result: ParseResult) -> None:
        """Extract ``require_once``, ``include``, etc. as import references.

        Handles simple string paths like ``require_once "config.php"`` and
        ``__DIR__ . "/helpers/utils.php"`` concatenation patterns.
        """
        path = self._extract_include_path(node)
        if not path:
            return

        # Normalize: strip leading ./ and resolve relative markers
        path = path.lstrip("./")

        result.imports.append(
            ImportInfo(
                module=path,
                names=[],
                is_relative=True,
            )
        )

    @staticmethod
    def _extract_include_path(node: Node) -> str:
        """Extract the file path string from an include/require node."""
        for child in node.children:
            if child.type == "encapsed_string":
                for sub in child.children:
                    if sub.type == "string_content":
                        return sub.text.decode()
            elif child.type == "string":
                for sub in child.children:
                    if sub.type == "string_content":
                        return sub.text.decode()
            elif child.type == "binary_expression":
                # Handle __DIR__ . "/path/to/file.php"
                for sub in child.children:
                    if sub.type == "encapsed_string":
                        for inner in sub.children:
                            if inner.type == "string_content":
                                return inner.text.decode()
                    elif sub.type == "string":
                        for inner in sub.children:
                            if inner.type == "string_content":
                                return inner.text.decode()
        return ""

    @staticmethod
    def _extract_identifier_arguments(call_node: Node) -> list[str]:
        """Extract bare identifier/variable arguments from a call."""
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            if child.type == "argument":
                for sub in child.children:
                    if sub.type == "variable_name":
                        text = sub.text.decode().lstrip("$")
                        identifiers.append(text)
                    elif sub.type == "name":
                        identifiers.append(sub.text.decode())
        return identifiers
