"""Dedicated PHP parser unit tests.

Covers all symbol extraction, import/use resolution, call extraction,
type reference extraction, heritage chains, trait support, enum support,
string method refs, variable type inference, and include path handling.
"""

from __future__ import annotations

import pytest

from axon.core.parsers.php import PhpParser


@pytest.fixture
def parser() -> PhpParser:
    return PhpParser()


# ---------------------------------------------------------------------------
# Symbol extraction — functions
# ---------------------------------------------------------------------------


class TestFunctionExtraction:
    """Top-level function definitions."""

    def test_simple_function(self, parser: PhpParser) -> None:
        code = "<?php\nfunction greet() { echo 'hi'; }"
        result = parser.parse(code, "test.php")
        funcs = [s for s in result.symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "greet"
        assert funcs[0].start_line == 2
        assert funcs[0].end_line == 2

    def test_function_with_params_and_return_type(self, parser: PhpParser) -> None:
        code = """<?php
function add(int $a, int $b): int {
    return $a + $b;
}
"""
        result = parser.parse(code, "math.php")
        funcs = [s for s in result.symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "add"
        assert "function add" in funcs[0].signature

    def test_multiple_functions(self, parser: PhpParser) -> None:
        code = """<?php
function foo() {}
function bar() {}
function baz() {}
"""
        result = parser.parse(code, "test.php")
        names = [s.name for s in result.symbols if s.kind == "function"]
        assert names == ["foo", "bar", "baz"]


# ---------------------------------------------------------------------------
# Symbol extraction — classes
# ---------------------------------------------------------------------------


class TestClassExtraction:
    """Class declarations with heritage."""

    def test_simple_class(self, parser: PhpParser) -> None:
        code = "<?php\nclass User {}"
        result = parser.parse(code, "user.php")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "User"

    def test_class_extends(self, parser: PhpParser) -> None:
        code = "<?php\nclass Admin extends User {}"
        result = parser.parse(code, "admin.php")
        assert ("Admin", "extends", "User") in result.heritage

    def test_class_implements(self, parser: PhpParser) -> None:
        code = "<?php\nclass UserRepo implements Repository, Cacheable {}"
        result = parser.parse(code, "repo.php")
        assert ("UserRepo", "implements", "Repository") in result.heritage
        assert ("UserRepo", "implements", "Cacheable") in result.heritage

    def test_class_extends_and_implements(self, parser: PhpParser) -> None:
        code = "<?php\nclass UserRepo extends BaseRepo implements Repository {}"
        result = parser.parse(code, "repo.php")
        assert ("UserRepo", "extends", "BaseRepo") in result.heritage
        assert ("UserRepo", "implements", "Repository") in result.heritage


# ---------------------------------------------------------------------------
# Symbol extraction — methods
# ---------------------------------------------------------------------------


class TestMethodExtraction:
    """Method declarations inside classes, interfaces, and traits."""

    def test_class_method(self, parser: PhpParser) -> None:
        code = """<?php
class User {
    public function getName(): string {
        return $this->name;
    }
}
"""
        result = parser.parse(code, "user.php")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "getName"
        assert methods[0].class_name == "User"

    def test_static_method(self, parser: PhpParser) -> None:
        code = """<?php
class Utils {
    public static function format($value): string {
        return (string) $value;
    }
}
"""
        result = parser.parse(code, "utils.php")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "format"
        assert methods[0].class_name == "Utils"

    def test_interface_method(self, parser: PhpParser) -> None:
        code = """<?php
interface Repository {
    public function find(int $id): ?object;
    public function save(object $entity): void;
}
"""
        result = parser.parse(code, "repo.php")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 2
        names = {m.name for m in methods}
        assert names == {"find", "save"}

    def test_trait_method_has_class_name(self, parser: PhpParser) -> None:
        code = """<?php
trait Loggable {
    public function log(string $msg): void {}
}
"""
        result = parser.parse(code, "loggable.php")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].class_name == "Loggable"


# ---------------------------------------------------------------------------
# Symbol extraction — interfaces
# ---------------------------------------------------------------------------


class TestInterfaceExtraction:
    """Interface declarations."""

    def test_simple_interface(self, parser: PhpParser) -> None:
        code = "<?php\ninterface Cacheable {}"
        result = parser.parse(code, "cache.php")
        ifaces = [s for s in result.symbols if s.kind == "interface"]
        assert len(ifaces) == 1
        assert ifaces[0].name == "Cacheable"

    def test_interface_extends(self, parser: PhpParser) -> None:
        code = "<?php\ninterface ReadWriteRepo extends ReadRepo {}"
        result = parser.parse(code, "repo.php")
        assert ("ReadWriteRepo", "extends", "ReadRepo") in result.heritage


# ---------------------------------------------------------------------------
# Symbol extraction — enums (PHP 8.1+)
# ---------------------------------------------------------------------------


class TestEnumExtraction:
    """PHP 8.1 enum declarations."""

    def test_basic_enum(self, parser: PhpParser) -> None:
        code = """<?php
enum Status {
    case Active;
    case Inactive;
}
"""
        result = parser.parse(code, "status.php")
        enums = [s for s in result.symbols if s.kind == "enum"]
        assert len(enums) == 1
        assert enums[0].name == "Status"

    def test_backed_enum(self, parser: PhpParser) -> None:
        code = """<?php
enum Color: string {
    case Red = 'red';
    case Blue = 'blue';
}
"""
        result = parser.parse(code, "color.php")
        enums = [s for s in result.symbols if s.kind == "enum"]
        assert len(enums) == 1
        assert enums[0].name == "Color"


# ---------------------------------------------------------------------------
# Symbol extraction — traits
# ---------------------------------------------------------------------------


class TestTraitExtraction:
    """Trait declarations and `use Trait;` heritage."""

    def test_trait_as_class_symbol(self, parser: PhpParser) -> None:
        code = """<?php
trait Auditable {
    public function auditLog(): array { return []; }
}
"""
        result = parser.parse(code, "audit.php")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert any(c.name == "Auditable" for c in classes)

    def test_use_trait_creates_heritage(self, parser: PhpParser) -> None:
        code = """<?php
trait Loggable {}
class Service {
    use Loggable;
}
"""
        result = parser.parse(code, "service.php")
        assert ("Service", "extends", "Loggable") in result.heritage

    def test_use_multiple_traits(self, parser: PhpParser) -> None:
        code = """<?php
class Model {
    use Loggable, Cacheable, Auditable;
}
"""
        result = parser.parse(code, "model.php")
        assert ("Model", "extends", "Loggable") in result.heritage
        assert ("Model", "extends", "Cacheable") in result.heritage
        assert ("Model", "extends", "Auditable") in result.heritage

    def test_use_qualified_trait(self, parser: PhpParser) -> None:
        code = r"""<?php
class Foo {
    use App\Traits\HasTimestamps;
}
"""
        result = parser.parse(code, "foo.php")
        trait_names = [h[2] for h in result.heritage]
        assert "HasTimestamps" in trait_names


# ---------------------------------------------------------------------------
# Imports — namespace use declarations
# ---------------------------------------------------------------------------


class TestImportExtraction:
    """PHP `use` statement (namespace import) extraction."""

    def test_simple_use(self, parser: PhpParser) -> None:
        code = r"<?php use App\Models\User;"
        result = parser.parse(code, "test.php")
        assert len(result.imports) == 1
        assert result.imports[0].names == ["User"]
        assert "User" in result.imports[0].module

    def test_aliased_use(self, parser: PhpParser) -> None:
        code = r"<?php use App\Models\User as AppUser;"
        result = parser.parse(code, "test.php")
        # When aliased, names contains the alias and alias field is set.
        imp = result.imports[0]
        assert imp.alias == "AppUser"
        # names may contain either the alias or the original; just verify import exists.
        assert "AppUser" in imp.names or "User" in imp.names

    def test_grouped_use(self, parser: PhpParser) -> None:
        code = r"""<?php
use App\Models\User;
use App\Models\Role;
use App\Models\Permission;
"""
        result = parser.parse(code, "test.php")
        all_names = []
        for imp in result.imports:
            all_names.extend(imp.names)
        assert "User" in all_names
        assert "Role" in all_names
        assert "Permission" in all_names


# ---------------------------------------------------------------------------
# Imports — require / include
# ---------------------------------------------------------------------------


class TestIncludeExtraction:
    """PHP require/include statement extraction."""

    def test_require_once_string(self, parser: PhpParser) -> None:
        code = '<?php require_once "config.php";'
        result = parser.parse(code, "app.php")
        assert len(result.imports) == 1
        assert "config.php" in result.imports[0].module

    def test_include_single_quoted(self, parser: PhpParser) -> None:
        code = "<?php include 'helpers.php';"
        result = parser.parse(code, "app.php")
        assert len(result.imports) == 1

    def test_require_dir_concat(self, parser: PhpParser) -> None:
        """__DIR__ . '/path' concatenation extracts the path portion."""
        code = """<?php require_once __DIR__ . '/helpers/utils.php';"""
        result = parser.parse(code, "app.php")
        assert len(result.imports) == 1
        assert "helpers/utils.php" in result.imports[0].module

    def test_include_relative_prefix_stripped(self, parser: PhpParser) -> None:
        code = '<?php require_once "./config.php";'
        result = parser.parse(code, "app.php")
        assert len(result.imports) == 1
        assert result.imports[0].module == "config.php"


# ---------------------------------------------------------------------------
# Call extraction
# ---------------------------------------------------------------------------


class TestCallExtraction:
    """Function, method, static, and new expression call extraction."""

    def test_function_call(self, parser: PhpParser) -> None:
        code = "<?php echo greet();"
        result = parser.parse(code, "test.php")
        call_names = [c.name for c in result.calls]
        assert "greet" in call_names

    def test_method_call(self, parser: PhpParser) -> None:
        code = '<?php $user->getName();'
        result = parser.parse(code, "test.php")
        calls = [(c.name, c.receiver) for c in result.calls]
        assert ("getName", "user") in calls

    def test_this_call(self, parser: PhpParser) -> None:
        code = """<?php
class Foo {
    public function bar() { $this->baz(); }
    private function baz() {}
}
"""
        result = parser.parse(code, "foo.php")
        calls = [(c.name, c.receiver) for c in result.calls]
        assert ("baz", "this") in calls

    def test_static_call(self, parser: PhpParser) -> None:
        code = "<?php User::find(1);"
        result = parser.parse(code, "test.php")
        calls = [(c.name, c.receiver) for c in result.calls]
        assert ("find", "User") in calls

    def test_new_expression(self, parser: PhpParser) -> None:
        code = '<?php $user = new User("John");'
        result = parser.parse(code, "test.php")
        call_names = [c.name for c in result.calls]
        assert "User" in call_names

    def test_callback_function_string(self, parser: PhpParser) -> None:
        """register_shutdown_function('handler') emits a call to 'handler'."""
        code = "<?php register_shutdown_function('myHandler');"
        result = parser.parse(code, "test.php")
        call_names = [c.name for c in result.calls]
        assert "myHandler" in call_names

    def test_nested_calls(self, parser: PhpParser) -> None:
        code = "<?php foo(bar(baz()));"
        result = parser.parse(code, "test.php")
        call_names = [c.name for c in result.calls]
        assert "foo" in call_names
        assert "bar" in call_names
        assert "baz" in call_names


# ---------------------------------------------------------------------------
# Type reference extraction
# ---------------------------------------------------------------------------


class TestTypeRefExtraction:
    """Parameter and return type annotations."""

    def test_param_type(self, parser: PhpParser) -> None:
        code = """<?php
function process(UserService $service): void {}
"""
        result = parser.parse(code, "test.php")
        param_refs = [t for t in result.type_refs if t.kind == "param"]
        assert any(t.name == "UserService" for t in param_refs)

    def test_return_type(self, parser: PhpParser) -> None:
        code = """<?php
function getUser(): User { return new User(); }
"""
        result = parser.parse(code, "test.php")
        return_refs = [t for t in result.type_refs if t.kind == "return"]
        assert any(t.name == "User" for t in return_refs)

    def test_builtin_type_not_extracted(self, parser: PhpParser) -> None:
        code = """<?php
function add(int $a, string $b): bool { return true; }
"""
        result = parser.parse(code, "test.php")
        # Built-in types should NOT appear in type_refs.
        names = {t.name for t in result.type_refs}
        assert "int" not in names
        assert "string" not in names
        assert "bool" not in names

    def test_union_return_type(self, parser: PhpParser) -> None:
        code = """<?php
function getClient(): LinqClient|BlueBubblesClient {
    return new LinqClient();
}
"""
        result = parser.parse(code, "test.php")
        return_refs = [t for t in result.type_refs if t.kind == "return"]
        names = {t.name for t in return_refs}
        assert "LinqClient" in names
        assert "BlueBubblesClient" in names

    def test_nullable_type(self, parser: PhpParser) -> None:
        code = """<?php
function findUser(int $id): ?User { return null; }
"""
        result = parser.parse(code, "test.php")
        return_refs = [t for t in result.type_refs if t.kind == "return"]
        assert any(t.name == "User" for t in return_refs)


# ---------------------------------------------------------------------------
# Variable type inference
# ---------------------------------------------------------------------------


class TestVariableTypeInference:
    """$var = func() and $var = new Class() tracking."""

    def test_new_expression_type(self, parser: PhpParser) -> None:
        code = """<?php
$client = new LinqClient();
"""
        result = parser.parse(code, "test.php")
        assert "client" in result.variable_types
        assert result.variable_types["client"] == ["LinqClient"]

    def test_factory_call_same_file(self, parser: PhpParser) -> None:
        code = """<?php
function getClient(): UserService {
    return new UserService();
}
$svc = getClient();
"""
        result = parser.parse(code, "test.php")
        assert "svc" in result.variable_types
        assert result.variable_types["svc"] == ["UserService"]

    def test_factory_cross_file_sentinel(self, parser: PhpParser) -> None:
        """Cross-file factory stores __call__ sentinel for later resolution."""
        code = """<?php
$client = getMessagingClient();
"""
        result = parser.parse(code, "test.php")
        assert "client" in result.variable_types
        assert result.variable_types["client"] == ["__call__getMessagingClient"]

    def test_non_variable_lhs_ignored(self, parser: PhpParser) -> None:
        code = """<?php
$this->client = new Client();
"""
        result = parser.parse(code, "test.php")
        # Property assignments should NOT be tracked (LHS is member, not variable)
        assert "client" not in result.variable_types


# ---------------------------------------------------------------------------
# String method refs (dynamic dispatch)
# ---------------------------------------------------------------------------


class TestStringMethodRefs:
    """String values in class property arrays emitting synthetic calls."""

    def test_string_matching_method_emits_call(self, parser: PhpParser) -> None:
        code = """<?php
class Fixer {
    private $patterns = ['callback' => 'fixBug'];
    public function apply() {}
    private function fixBug() {}
}
"""
        result = parser.parse(code, "fixer.php")
        calls = [(c.name, c.receiver) for c in result.calls]
        assert ("fixBug", "this") in calls

    def test_non_method_string_not_emitted(self, parser: PhpParser) -> None:
        code = """<?php
class Config {
    private $settings = ['name' => 'MyApp', 'version' => '1.0'];
    public function get() { return $this->settings; }
}
"""
        result = parser.parse(code, "config.php")
        # "MyApp" and "1.0" should NOT be emitted as calls.
        call_names = [c.name for c in result.calls]
        assert "MyApp" not in call_names
        assert "1.0" not in call_names

    def test_dunder_string_excluded(self, parser: PhpParser) -> None:
        code = """<?php
class Foo {
    private $hooks = ['__construct'];
    public function __construct() {}
}
"""
        result = parser.parse(code, "foo.php")
        synthetic = [c for c in result.calls if c.name == "__construct" and c.receiver == "this"]
        assert len(synthetic) == 0


# ---------------------------------------------------------------------------
# Namespace declarations
# ---------------------------------------------------------------------------


class TestNamespaceExtraction:
    """PHP namespace declaration tracking."""

    def test_namespace_stored_on_symbols(self, parser: PhpParser) -> None:
        code = r"""<?php
namespace App\Services;

class UserService {
    public function find() {}
}
"""
        result = parser.parse(code, "UserService.php")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].namespace == r"App\Services"

    def test_namespace_on_function(self, parser: PhpParser) -> None:
        code = r"""<?php
namespace App\Helpers;

function formatDate($date) { return $date; }
"""
        result = parser.parse(code, "helpers.php")
        funcs = [s for s in result.symbols if s.kind == "function"]
        assert len(funcs) == 1
        assert funcs[0].namespace == r"App\Helpers"

    def test_no_namespace(self, parser: PhpParser) -> None:
        code = "<?php\nfunction foo() {}"
        result = parser.parse(code, "test.php")
        funcs = [s for s in result.symbols if s.kind == "function"]
        assert funcs[0].namespace == ""

    def test_namespace_on_methods(self, parser: PhpParser) -> None:
        code = r"""<?php
namespace App\Models;

class User {
    public function getName() { return ''; }
}
"""
        result = parser.parse(code, "User.php")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].namespace == r"App\Models"


# ---------------------------------------------------------------------------
# Include path extraction edge cases
# ---------------------------------------------------------------------------


class TestIncludePathEdgeCases:
    """Edge cases in PHP include/require path extraction."""

    def test_dir_parent_path(self, parser: PhpParser) -> None:
        """__DIR__ . '/../config.php' extracts parent path."""
        code = """<?php require_once __DIR__ . '/../config/config.php';"""
        result = parser.parse(code, "app.php")
        assert len(result.imports) == 1
        assert "config/config.php" in result.imports[0].module

    def test_double_quoted_include(self, parser: PhpParser) -> None:
        code = '<?php include_once __DIR__ . "/helpers.php";'
        result = parser.parse(code, "app.php")
        assert len(result.imports) == 1

    def test_no_path_no_import(self, parser: PhpParser) -> None:
        """Dynamic include with variable should not crash."""
        code = "<?php require_once $configPath;"
        result = parser.parse(code, "app.php")
        # Variable includes cannot be resolved — no import emitted.
        assert len(result.imports) == 0


# ---------------------------------------------------------------------------
# Heritage chains
# ---------------------------------------------------------------------------


class TestHeritageChains:
    """Complex class hierarchies."""

    def test_abstract_class_extends(self, parser: PhpParser) -> None:
        code = """<?php
abstract class BaseController {
    abstract public function index();
}
class UserController extends BaseController {
    public function index() {}
}
"""
        result = parser.parse(code, "controllers.php")
        assert ("UserController", "extends", "BaseController") in result.heritage

    def test_interface_chain(self, parser: PhpParser) -> None:
        code = """<?php
interface Readable {}
interface Writable extends Readable {}
class FileStore implements Writable {}
"""
        result = parser.parse(code, "store.php")
        assert ("Writable", "extends", "Readable") in result.heritage
        assert ("FileStore", "implements", "Writable") in result.heritage
