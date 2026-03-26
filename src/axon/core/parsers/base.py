"""Base parser interface and shared data structures.

Defines the intermediate representation produced by language-specific parsers
before the data is mapped into the knowledge graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SymbolInfo:
    """A parsed symbol (function, class, method, etc.)."""

    name: str
    kind: str  # "function", "class", "method", "interface", "type_alias", "enum"
    start_line: int
    end_line: int
    content: str
    signature: str = ""
    class_name: str = ""  # for methods: the owning class
    # e.g. ["staticmethod", "server.list_tools"]
    decorators: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """A parsed import statement.

    Contract:
    - ``module``: the source module path (e.g. ``"os.path"``, ``"./utils"``).
    - ``names``: the symbols being imported from *module* (e.g. ``["join", "exists"]``).
      For ``import numpy as np``, ``names=["numpy"]`` (the last segment of the module),
      NOT the alias.  For ``from os.path import join``, ``names=["join"]``.
    - ``alias``: the local binding name when the import is aliased
      (e.g. ``"np"`` for ``import numpy as np``, ``""`` otherwise).
      Import resolution uses ``module`` to locate the target file; ``alias`` is
      only relevant for local-name lookups by callers.
    """

    module: str  # the module path (e.g., "os.path", "./utils")
    names: list[str] = field(default_factory=list)  # imported names (e.g., ["join", "exists"])
    is_relative: bool = False
    alias: str = ""  # local binding name when aliased (e.g. "np" for "import numpy as np")


@dataclass
class CallInfo:
    """A parsed function call."""

    name: str  # the called function/method name
    line: int
    receiver: str = ""  # for method calls: the object (e.g., "self", "user")
    arguments: list[str] = field(default_factory=list)  # bare identifier arguments (callbacks)
    # variable name if result is assigned (e.g. "user" in "user = User()")
    assignment_target: str = ""


@dataclass
class TypeRef:
    """A parsed type annotation reference."""

    name: str  # the type name (e.g., "User", "list", "str")
    kind: str  # "param", "return", "variable"
    line: int
    param_name: str = ""  # for param types: the parameter name
    variable_name: str = ""  # for variable types: the annotated variable name


@dataclass
class ParseResult:
    """Complete parse result for a single file."""

    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)
    type_refs: list[TypeRef] = field(default_factory=list)
    heritage: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (class_name, kind, parent_name) where kind is "extends" or "implements"
    exports: list[str] = field(default_factory=list)  # names from __all__ or export statements
    func_refs: list[FuncRef] = field(default_factory=list)  # first-class function references


@dataclass
class FuncRef:
    """A first-class function reference (e.g. ``handler = my_func``)."""

    name: str  # the referenced function name
    line: int
    target_var: str = ""  # the variable being assigned to


class LanguageParser(ABC):
    """Base interface for language-specific parsers."""

    @abstractmethod
    def parse(self, content: str, file_path: str) -> ParseResult: ...
