"""TypeScript / TSX / JavaScript parser using tree-sitter.

Extracts symbols (functions, classes, methods, interfaces, type aliases),
imports, call expressions, type annotation references, and heritage
(extends / implements) relationships from TypeScript, TSX, and JavaScript
source files.
"""

from __future__ import annotations

import re

import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)

TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())
JS_LANGUAGE = Language(tsjavascript.language())

_DIALECT_MAP: dict[str, Language] = {
    "typescript": TS_LANGUAGE,
    "tsx": TSX_LANGUAGE,
    "javascript": JS_LANGUAGE,
}

_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "string",
        "number",
        "boolean",
        "void",
        "any",
        "unknown",
        "never",
        "null",
        "undefined",
        "object",
    }
)

class TypeScriptParser(LanguageParser):
    """Parse TypeScript, TSX, or JavaScript files via tree-sitter.

    Args:
        dialect: One of ``"typescript"``, ``"tsx"``, or ``"javascript"``.
    """

    def __init__(self, dialect: str = "typescript") -> None:
        if dialect not in _DIALECT_MAP:
            raise ValueError(
                f"Unknown dialect {dialect!r}. "
                f"Expected one of: {', '.join(sorted(_DIALECT_MAP))}"
            )
        self.dialect = dialect
        self._language = _DIALECT_MAP[dialect]
        self._parser = Parser(self._language)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse *content* and return an intermediate :class:`ParseResult`."""
        tree = self._parser.parse(content.encode("utf-8"))

        result = ParseResult()
        self._walk(tree.root_node, content, result)
        return result

    def _walk(
        self, node: Node, source: str, result: ParseResult, visited: set[int] | None = None
    ) -> None:
        """Walk the tree recursively, dispatching on node type.

        Uses a *visited* set (keyed by node ``id``) to avoid processing
        the same subtree twice — e.g. class bodies that are walked by both
        ``_extract_class`` and the generic child recursion.
        """
        if visited is None:
            visited = set()

        node_key = node.id
        if node_key in visited:
            return
        visited.add(node_key)

        ntype = node.type

        if ntype == "export_statement":
            self._extract_export(node, source, result)
        elif ntype == "function_declaration":
            self._extract_function_declaration(node, source, result)
        elif ntype in ("lexical_declaration", "variable_declaration"):
            self._extract_variable_declaration(node, source, result)
        elif ntype == "class_declaration":
            self._extract_class(node, source, result)
        elif ntype == "interface_declaration":
            self._extract_interface(node, source, result)
        elif ntype == "type_alias_declaration":
            self._extract_type_alias(node, source, result)
        elif ntype == "import_statement":
            self._extract_import(node, source, result)
        elif ntype == "call_expression":
            self._extract_call(node, source, result)
        elif ntype == "new_expression":
            self._extract_new_expression(node, source, result)
        elif ntype == "expression_statement":
            self._maybe_extract_module_exports(node, source, result)
        elif ntype == "method_definition":
            self._extract_method(node, source, result)
        elif ntype in ("jsx_opening_element", "jsx_self_closing_element"):
            self._extract_jsx_callbacks(node, result)
        elif ntype == "template_string":
            self._extract_template_onclick_refs(node, result)
        elif ntype == "shorthand_property_identifier":
            # { funcA, funcB } — shorthand object property is a reference
            # to a same-scope symbol (function, variable, etc.).
            result.calls.append(
                CallInfo(name=node.text.decode(), line=node.start_point[0] + 1)
            )

        for child in node.children:
            self._walk(child, source, result, visited)

    def _extract_export(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Handle ``export`` statements — mark exported symbol names.

        Handles ``export function foo()``, ``export class Bar``,
        ``export const baz = ...``, and ``export { name1, name2 }``.
        """
        for child in node.children:
            if child.type in (
                "function_declaration",
                "class_declaration",
                "interface_declaration",
                "type_alias_declaration",
            ):
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    result.exports.append(name_node.text.decode())
            elif child.type in ("lexical_declaration", "variable_declaration"):
                for sub in child.children:
                    if sub.type == "variable_declarator":
                        name_node = sub.child_by_field_name("name")
                        if name_node is not None:
                            result.exports.append(name_node.text.decode())
            elif child.type == "export_clause":
                # export { name1, name2 }
                for spec in child.children:
                    if spec.type == "export_specifier":
                        name_node = spec.child_by_field_name("name")
                        if name_node is not None:
                            result.exports.append(name_node.text.decode())

    def _maybe_extract_module_exports(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Handle ``module.exports``, ``exports``, and ``window.X`` assignments.

        Detects three patterns that expose symbols to other files:
        - ``module.exports = X`` / ``module.exports = { A, B }``
        - ``exports.X = ...``
        - ``window.MyNamespace = { A, B, C }`` (browser-global revealing pattern)
        """
        for child in node.children:
            if child.type != "assignment_expression":
                continue
            left = child.child_by_field_name("left")
            right = child.child_by_field_name("right")
            if left is None or right is None:
                continue

            left_text = left.text.decode()

            is_module_export = left_text in ("module.exports", "exports")
            is_window_export = (
                left.type == "member_expression"
                and left_text.startswith("window.")
            )

            if not is_module_export and not is_window_export:
                continue

            if right.type == "identifier":
                result.exports.append(right.text.decode())
            elif right.type == "object":
                for prop in right.children:
                    if prop.type == "shorthand_property_identifier":
                        result.exports.append(prop.text.decode())
                    elif prop.type == "pair":
                        key_node = prop.child_by_field_name("key")
                        if key_node is not None:
                            result.exports.append(key_node.text.decode())

    def _extract_function_declaration(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()
        signature = self._build_function_signature(node, name)

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

        self._extract_function_types(node, name, result)

    def _extract_method(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract a method_definition inside a class body."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        class_name = self._find_parent_class_name(node)

        signature = self._build_function_signature(node, name)

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

        self._extract_function_types(node, name, result)

    def _extract_variable_declaration(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Handle arrow functions, function expressions, and require() calls."""
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if name_node is None or value_node is None:
                continue

            var_name = name_node.text.decode()

            if value_node.type in ("arrow_function", "function_expression"):
                self._extract_assigned_function(child, var_name, value_node, result)
            elif value_node.type == "call_expression":
                self._maybe_extract_require(child, var_name, value_node, result)

            self._extract_variable_type_annotation(child, result)

    def _extract_assigned_function(
        self,
        declarator_node: Node,
        name: str,
        func_node: Node,
        result: ParseResult,
    ) -> None:
        """Extract an arrow function or function expression assigned to a variable."""
        outer = declarator_node.parent
        if outer is None:
            outer = declarator_node

        start_line = outer.start_point[0] + 1
        end_line = outer.end_point[0] + 1
        content = outer.text.decode()
        signature = self._build_function_signature(func_node, name)

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

        self._extract_function_types(func_node, name, result)

    def _maybe_extract_require(
        self,
        declarator_node: Node,
        var_name: str,
        call_node: Node,
        result: ParseResult,
    ) -> None:
        """If the call is ``require('./foo')``, emit an ImportInfo."""
        func_node = call_node.child_by_field_name("function")
        if func_node is None or func_node.text.decode() != "require":
            return

        args = call_node.child_by_field_name("arguments")
        if args is None:
            return

        module_str = ""
        for arg_child in args.children:
            if arg_child.type == "string":
                module_str = self._string_value(arg_child)
                break

        if not module_str:
            return

        result.imports.append(
            ImportInfo(
                module=module_str,
                names=[var_name],
                is_relative=module_str.startswith("."),
            )
        )

    def _extract_class(self, node: Node, source: str, result: ParseResult) -> None:
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
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        for child in node.children:
            if child.type == "class_heritage":
                self._extract_class_heritage(name, child, result)

    def _extract_class_heritage(
        self, class_name: str, heritage_node: Node, result: ParseResult
    ) -> None:
        for child in heritage_node.children:
            if child.type == "extends_clause":
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        result.heritage.append((class_name, "extends", sub.text.decode()))
            elif child.type == "implements_clause":
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        result.heritage.append((class_name, "implements", sub.text.decode()))

    def _extract_interface(self, node: Node, source: str, result: ParseResult) -> None:
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

        for child in node.children:
            if child.type == "extends_type_clause":
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        result.heritage.append((name, "extends", sub.text.decode()))

    def _extract_type_alias(self, node: Node, source: str, result: ParseResult) -> None:
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
                kind="type_alias",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

    def _extract_import(self, node: Node, source: str, result: ParseResult) -> None:
        """Handle ES module import statements."""
        module_str = ""
        names: list[str] = []
        alias = ""

        source_node = node.child_by_field_name("source")
        if source_node is not None:
            module_str = self._string_value(source_node)
        else:
            # Fallback: look for a string child after 'from'.
            for child in node.children:
                if child.type == "string":
                    module_str = self._string_value(child)
                    break

        if not module_str:
            return

        import_clause = None
        for child in node.children:
            if child.type == "import_clause":
                import_clause = child
                break

        if import_clause is not None:
            for clause_child in import_clause.children:
                if clause_child.type == "named_imports":
                    # import { A, B } from '...'
                    for spec in clause_child.children:
                        if spec.type == "import_specifier":
                            name_node = spec.child_by_field_name("name")
                            if name_node is not None:
                                names.append(name_node.text.decode())
                elif clause_child.type == "namespace_import":
                    # import * as utils from '...'
                    for ns_child in clause_child.children:
                        if ns_child.type == "identifier":
                            alias = ns_child.text.decode()
                            names.append(alias)
                            break
                elif clause_child.type == "identifier":
                    # import Foo from '...'  (default import)
                    names.append(clause_child.text.decode())

        result.imports.append(
            ImportInfo(
                module=module_str,
                names=names,
                is_relative=module_str.startswith("."),
                alias=alias,
            )
        )

    def _extract_call(self, node: Node, source: str, result: ParseResult) -> None:
        func_node = node.child_by_field_name("function")
        if func_node is None:
            return

        line = node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(node)

        if func_node.type == "member_expression":
            obj_node = func_node.child_by_field_name("object")
            prop_node = func_node.child_by_field_name("property")
            if prop_node is not None:
                receiver = obj_node.text.decode() if obj_node else ""
                prop_text = prop_node.text.decode()
                result.calls.append(
                    CallInfo(
                        name=prop_text,
                        line=line,
                        receiver=receiver,
                        arguments=arguments,
                    )
                )
                # React.createElement(Component, {onClick: handler})
                if prop_text == "createElement":
                    self._extract_create_element_refs(node, result)
        elif func_node.type == "identifier":
            name = func_node.text.decode()
            # Skip require() since it's handled as an import.
            if name != "require":
                result.calls.append(CallInfo(name=name, line=line, arguments=arguments))

    def _extract_new_expression(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Handle ``new ClassName(args)`` — emit a CallInfo targeting the class."""
        constructor_node = node.child_by_field_name("constructor")
        if constructor_node is None:
            return

        line = node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(node)

        if constructor_node.type == "identifier":
            result.calls.append(
                CallInfo(
                    name=constructor_node.text.decode(),
                    line=line,
                    arguments=arguments,
                )
            )
        elif constructor_node.type == "member_expression":
            obj_node = constructor_node.child_by_field_name("object")
            prop_node = constructor_node.child_by_field_name("property")
            if prop_node is not None:
                receiver = obj_node.text.decode() if obj_node else ""
                result.calls.append(
                    CallInfo(
                        name=prop_node.text.decode(),
                        line=line,
                        receiver=receiver,
                        arguments=arguments,
                    )
                )

    def _extract_jsx_callbacks(self, node: Node, result: ParseResult) -> None:
        """Extract component references and callback props from JSX elements.

        1. If the tag name starts with an uppercase letter it is a React
           component — emit a CallInfo so the component gets a CALLS edge.
        2. For any ``on`` + uppercase attribute (e.g. ``onClick``,
           ``onSave``), extract bare identifiers and member expressions
           as callback references.
        """
        line = node.start_point[0] + 1

        # --- Component tag name ---
        for child in node.children:
            if child.type == "identifier":
                tag_name = child.text.decode()
                if tag_name and tag_name[0].isupper():
                    result.calls.append(CallInfo(name=tag_name, line=line))
                break
            elif child.type == "member_expression":
                prop = child.child_by_field_name("property")
                obj = child.child_by_field_name("object")
                if prop is not None:
                    result.calls.append(
                        CallInfo(
                            name=prop.text.decode(),
                            line=line,
                            receiver=obj.text.decode() if obj else "",
                        )
                    )
                break

        # --- Prop function references ---
        # Any JSX attribute whose value is a bare identifier or member
        # expression is a potential function reference (not just on* props).
        # This captures patterns like ``viewCampaign={viewCampaign}`` and
        # ``closeCampaignDetail={closeCampaignDetail}``.
        for child in node.children:
            if child.type != "jsx_attribute":
                continue
            attr_value_node = None
            for sub in child.children:
                if sub.type == "jsx_expression":
                    attr_value_node = sub

            if attr_value_node is None:
                continue

            for expr_child in attr_value_node.children:
                if expr_child.type == "identifier":
                    result.calls.append(
                        CallInfo(
                            name=expr_child.text.decode(),
                            line=child.start_point[0] + 1,
                        )
                    )
                elif expr_child.type == "member_expression":
                    prop = expr_child.child_by_field_name("property")
                    obj = expr_child.child_by_field_name("object")
                    if prop is not None:
                        receiver = obj.text.decode() if obj else ""
                        result.calls.append(
                            CallInfo(
                                name=prop.text.decode(),
                                line=child.start_point[0] + 1,
                                receiver=receiver,
                            )
                        )

    def _extract_create_element_refs(self, node: Node, result: ParseResult) -> None:
        """Extract component and callback refs from ``React.createElement`` calls.

        ``React.createElement(Component, {onClick: handler})`` — the first
        argument is a component reference (if uppercase identifier) and the
        second argument may contain ``on*`` callback properties.
        """
        args_node = node.child_by_field_name("arguments")
        if args_node is None:
            return

        line = node.start_point[0] + 1
        arg_children = [
            c for c in args_node.children if c.type not in (",", "(", ")")
        ]

        # First argument: component name (uppercase identifier = React component)
        if arg_children:
            first_arg = arg_children[0]
            if first_arg.type == "identifier":
                name = first_arg.text.decode()
                if name and name[0].isupper():
                    result.calls.append(CallInfo(name=name, line=line))

        # Second argument: props object — extract identifier values as potential
        # function references (same logic as expanded JSX prop handling).
        if len(arg_children) >= 2:
            props_arg = arg_children[1]
            if props_arg.type == "object":
                for prop in props_arg.children:
                    if prop.type == "shorthand_property_identifier":
                        result.calls.append(
                            CallInfo(name=prop.text.decode(), line=line)
                        )
                    elif prop.type == "pair":
                        value = prop.child_by_field_name("value")
                        if value is not None and value.type == "identifier":
                            result.calls.append(
                                CallInfo(name=value.text.decode(), line=line)
                            )

    # Matches function references inside template literal strings:
    # 1. onclick="funcName(..." — inline event handler in generated HTML
    # 2. `funcName(${arg})` — bare function call string (used for dynamic onclick)
    _TEMPLATE_EVENT_RE = re.compile(
        r"""on\w+\s*=\s*["']([A-Za-z_$][A-Za-z0-9_$]*)\s*\(""",
    )
    _TEMPLATE_BARE_CALL_RE = re.compile(
        r"""^([A-Za-z_$][A-Za-z0-9_$]*)\s*\(""",
    )

    def _extract_template_onclick_refs(self, node: Node, result: ParseResult) -> None:
        """Extract function references from inline event handlers in template strings.

        Scans template literal content for patterns like ``onclick="funcName(..."``
        and emits synthetic CallInfo entries so the referenced function gets a
        CALLS edge instead of being flagged as dead code.
        """
        line = node.start_point[0] + 1
        for child in node.children:
            if child.type == "string_fragment" or child.type == "template_content":
                text = child.text.decode()
            elif child.type == "template_substitution":
                continue
            else:
                continue
            for m in self._TEMPLATE_EVENT_RE.finditer(text):
                func_name = m.group(1)
                result.calls.append(CallInfo(name=func_name, line=line))
            # Bare function call at start of template fragment
            # (e.g. `openCampaignStatsById(${c.id})`)
            m = self._TEMPLATE_BARE_CALL_RE.match(text.strip())
            if m:
                func_name = m.group(1)
                result.calls.append(CallInfo(name=func_name, line=line))

    @staticmethod
    def _extract_identifier_arguments(call_node: Node) -> list[str]:
        """Extract bare identifier arguments from a call_expression node."""
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            if child.type == "identifier":
                identifiers.append(child.text.decode())
        return identifiers

    def _extract_function_types(
        self, func_node: Node, func_name: str, result: ParseResult
    ) -> None:
        """Extract parameter types and return type from a function-like node."""
        params = func_node.child_by_field_name("parameters")
        if params is None:
            # Some nodes use "formal_parameters" via children iteration.
            for child in func_node.children:
                if child.type == "formal_parameters":
                    params = child
                    break

        if params is not None:
            for param in params.children:
                if param.type in ("required_parameter", "optional_parameter"):
                    param_name_node = param.child_by_field_name("name")
                    if param_name_node is None:
                        # Fallback: first identifier child.
                        for sub in param.children:
                            if sub.type == "identifier":
                                param_name_node = sub
                                break
                    if param_name_node is None:
                        continue

                    param_name = param_name_node.text.decode()

                    for sub in param.children:
                        if sub.type == "type_annotation":
                            type_name = self._type_annotation_name(sub)
                            if type_name and type_name.lower() not in _BUILTIN_TYPES:
                                result.type_refs.append(
                                    TypeRef(
                                        name=type_name,
                                        kind="param",
                                        line=sub.start_point[0] + 1,
                                        param_name=param_name,
                                    )
                                )

        # Return type: type_annotation directly on the function node (not inside params).
        for child in func_node.children:
            if child.type == "type_annotation":
                type_name = self._type_annotation_name(child)
                if type_name and type_name.lower() not in _BUILTIN_TYPES:
                    result.type_refs.append(
                        TypeRef(
                            name=type_name,
                            kind="return",
                            line=child.start_point[0] + 1,
                        )
                    )

    def _extract_variable_type_annotation(
        self, declarator_node: Node, result: ParseResult
    ) -> None:
        """Extract type from ``const x: Config = ...``."""
        for child in declarator_node.children:
            if child.type == "type_annotation":
                type_name = self._type_annotation_name(child)
                if type_name and type_name.lower() not in _BUILTIN_TYPES:
                    result.type_refs.append(
                        TypeRef(
                            name=type_name,
                            kind="variable",
                            line=child.start_point[0] + 1,
                        )
                    )

    @staticmethod
    def _type_annotation_name(annotation_node: Node) -> str:
        """Return the simple type name from a ``type_annotation`` node.

        Handles ``type_identifier``, ``predefined_type``, and ``identifier``
        children.  For compound types (unions, generics, etc.) returns the
        text of the first recognisable child.
        """
        for child in annotation_node.children:
            if child.type in ("type_identifier", "predefined_type", "identifier"):
                return child.text.decode()
        return ""

    @staticmethod
    def _string_value(string_node: Node) -> str:
        """Extract the raw string value from a tree-sitter ``string`` node.

        String nodes look like: string -> [quote, string_fragment, quote].
        """
        for child in string_node.children:
            if child.type == "string_fragment":
                return child.text.decode()
        # Fallback: strip outer quotes from the whole text.
        text = string_node.text.decode()
        if len(text) >= 2 and text[0] in ("'", '"', "`") and text[-1] in ("'", '"', "`"):
            return text[1:-1]
        return text

    @staticmethod
    def _build_function_signature(node: Node, name: str) -> str:
        """Build a human-readable signature line for a function-like node.

        Includes the parameter list and return type (if present).
        """
        params_text = ""
        return_type = ""

        for child in node.children:
            if child.type == "formal_parameters":
                params_text = child.text.decode()
            elif child.type == "type_annotation":
                return_type = child.text.decode()

        sig = f"{name}{params_text}"
        if return_type:
            sig += return_type
        return sig

    @staticmethod
    def _find_parent_class_name(node: Node) -> str:
        """Walk up the tree to find the enclosing class name."""
        current = node.parent
        while current is not None:
            if current.type == "class_declaration":
                name_node = current.child_by_field_name("name")
                if name_node is not None:
                    return name_node.text.decode()
            current = current.parent
        return ""
