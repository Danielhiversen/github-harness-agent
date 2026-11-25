"""Find a small set of static dependencies for a speculative pinned restore."""

import ast
import builtins
import symtable
from collections import Counter

from .review_definitions import _import_index, _symbols
from .review_line_ranges import _scope

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (*_FUNCTIONS, ast.ClassDef, ast.Lambda)
_IMPORTS = (ast.Import, ast.ImportFrom)
_CONTAINERS = {
    **{item: item for item in ("dict", "list", "set", "tuple", "frozenset")},
    **{
        f"typing.{item}": item.lower()
        for item in ("Dict", "List", "Set", "Tuple", "FrozenSet")
    },
}


def _walk(nodes):
    pending = list(nodes)
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, _SCOPES):
            pending.extend(ast.iter_child_nodes(node))


def _alias_name(alias):
    return alias.asname or alias.name.partition(".")[0]


def _binds(nodes):
    names = Counter()
    for node in _walk(nodes):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names[node.id] += 1
        elif isinstance(node, ast.alias):
            names[_alias_name(node)] += 1
        elif isinstance(
            node,
            (*_FUNCTIONS, ast.ClassDef, ast.MatchAs, ast.MatchStar, ast.ExceptHandler),
        ):
            names[node.name] += 1
        elif isinstance(node, ast.MatchMapping):
            names[node.rest] += 1
    return names


def _container(annotation):
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except (SyntaxError, ValueError, RecursionError):
            return None
    if isinstance(annotation, ast.Subscript):
        annotation = annotation.value
    return _CONTAINERS.get(ast.unparse(annotation))


def _direct_helpers(scope):
    args = (*scope.node.args.posonlyargs, *scope.node.args.args)
    if (
        not args
        or args[0].arg != "self"
        or scope.node.decorator_list
        or _binds(scope.node.body)["self"]
    ):
        return set()
    return {
        node.func.attr
        for node in _walk(scope.node.body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def _unambiguous_helper(scope, owner, name):
    cls = next(
        (
            node
            for qualified, node in _symbols(scope.tree.body)
            if qualified == owner and isinstance(node, ast.ClassDef)
        ),
        None,
    )
    if cls is None:
        return False
    bindings = _binds(cls.body)
    if bindings[name] != 1 or bindings["__getattribute__"]:
        return False
    for node in ast.walk(scope.tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return False
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in {"self", cls.name}
            and (
                not isinstance(node.args[1], ast.Constant) or node.args[1].value == name
            )
        ):
            return False
    return True


def _stable_decorators(before, after, left, right):
    old, new = left.decorator_list, right.decorator_list
    if not old and not new:
        return True
    if len(old) != 1 or len(new) != 1:
        return False
    name = getattr(old[0], "id", None)
    return (
        name in {"staticmethod", "classmethod"}
        and getattr(new[0], "id", None) == name
        and not any(_binds(scope.tree.body)[name] for scope in (before, after))
    )


def _changed_helpers(before, after, qualified):
    owner, sep, name = qualified.rpartition(".")
    if not sep:
        return [], False
    pinned, draft = "".join(before.rows), "".join(after.rows)
    old_defs, new_defs = (
        list(_symbols(before.tree.body)),
        list(_symbols(after.tree.body)),
    )
    matches, ambiguous = [], False
    for helper in sorted(_direct_helpers(before) - {name}):
        symbol = f"{owner}.{helper}"
        left_defs = [item for key, item in old_defs if key == symbol]
        right_defs = [item for key, item in new_defs if key == symbol]
        if len(left_defs) != 1 or len(right_defs) != 1:
            ambiguous |= len(left_defs) > 1 or len(right_defs) > 1
            continue
        old_node, new_node = left_defs[0], right_defs[0]
        if type(old_node) is not type(new_node) or not isinstance(old_node, _FUNCTIONS):
            continue
        old, new = _container(old_node.returns), _container(new_node.returns)
        if not old or not new or old == new:
            continue
        if not _stable_decorators(before, after, old_node, new_node) or not all(
            _unambiguous_helper(scope, owner, helper) for scope in (before, after)
        ):
            ambiguous = True
            continue
        left, right = _scope(pinned, symbol), _scope(draft, symbol)
        if left is not None and right is not None:
            matches.append((symbol, left, right, old, new))
        if len(matches) > 1:
            return matches, True
    return matches, ambiguous


def _global_reads(table, qualified, scope):
    for part in qualified.split("."):
        children = [child for child in table.get_children() if child.get_name() == part]
        if len(children) != 1:
            return set()
        table = children[0]
    reads = {
        entry.get_name()
        for entry in table.get_symbols()
        if entry.is_global() and entry.is_referenced()
    }
    # Annotations and defaults are evaluated outside the function's symbol table.
    header = [scope.node.args, *scope.node.decorator_list]
    if scope.node.returns is not None:
        header.append(scope.node.returns)
    owner = qualified.rpartition(".")[0]
    class_names = next(
        (
            _binds(node.body)
            for key, node in _symbols(scope.tree.body)
            if key == owner and isinstance(node, ast.ClassDef)
        ),
        {},
    )
    reads.update(
        node.id
        for node in _walk(header)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and not class_names.get(node.id)
    )
    return reads


def _dynamic_bindings(tree):
    return any(
        (
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
        )
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"exec", "eval", "globals"}
        )
        for node in ast.walk(tree)
    )


def _missing_imports(before, after, targets):
    if any(_dynamic_bindings(view.tree) for view in (before, after)):
        return [], []
    table = symtable.symtable("".join(before.rows), "<pinned stock source>", "exec")
    reads = set().union(
        *(_global_reads(table, name, left) for name, left, _ in targets)
    )
    pinned, current = _binds(before.tree.body), _binds(after.tree.body)
    written_globals = {
        name
        for scope in (before, after)
        for node in ast.walk(scope.tree)
        if isinstance(node, ast.Global)
        for name in node.names
    }
    outside = {
        node.id
        for node in ast.walk(after.tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and not any(
            right.start < node.lineno <= right.node.end_lineno
            for _, _, right in targets
        )
    }
    imports, conflicts = [], []
    import_end = _import_index(before.tree)
    for item in before.tree.body:
        if not isinstance(item, _IMPORTS) or item.end_lineno > import_end:
            continue
        if isinstance(item, ast.ImportFrom) and item.module == "__future__":
            continue
        for alias in item.names:
            name = _alias_name(alias)
            if name not in reads or pinned[name] != 1 or name in vars(builtins):
                continue
            if current[name] or name in written_globals or name in outside:
                conflicts.append(name)
                continue
            binding = (
                ast.ImportFrom(module=item.module, names=[alias], level=item.level)
                if isinstance(item, ast.ImportFrom)
                else ast.Import(names=[alias])
            )
            imports.append((name, ast.unparse(binding)))
    return imports, conflicts


def pinned_restore_dependencies(before, after, qualified):
    """Return exact pinned helper edits, missing import bindings and bounded facts."""
    targets = [(qualified, before, after)]
    edits, facts = [], []
    helpers, ambiguous = _changed_helpers(before, after, qualified)
    if ambiguous or len(helpers) > 1:
        facts.append("Changed direct helper bindings are ambiguous.")
    elif helpers:
        symbol, left, right, old, new = helpers[0]
        targets.append((symbol, left, right))
        code = "".join(left.rows[left.start : left.node.end_lineno])
        edits.append((right.start, right.node.end_lineno, code))
        facts.append(f"{symbol} declares pinned {old} and draft {new} returns.")
    imports, conflicts = _missing_imports(before, after, targets)
    if len(imports) > 2:
        facts.append("More than two absent import bindings were found.")
    elif imports:
        index = _import_index(after.tree)
        block = "".join(statement + after.newline for _, statement in imports)
        edits.append((index, index, block))
        facts.append(
            "Absent pinned import bindings: "
            + ", ".join(name for name, _ in imports)
            + "."
        )
    if conflicts:
        facts.append(
            "Existing or shared draft bindings were left intact: "
            + ", ".join(conflicts[:2])
            + "."
        )
    return edits, " ".join(facts)
