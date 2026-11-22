"""Identify an exercised keyword that ceased to bound iteration in a review edit."""

import ast
from textwrap import dedent

from .review_definitions import _symbols


def _reads(node, name):
    return {
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
        and item.id == name
        and isinstance(item.ctx, ast.Load)
    }


def lost_iteration_argument(fixture, pinned, current):
    try:
        calls = {
            (
                call.func.attr
                if isinstance(call.func, ast.Attribute)
                else call.func.id,
                kw.arg,
            )
            for call in ast.walk(ast.parse(dedent(fixture)))
            if isinstance(call, ast.Call)
            and isinstance(call.func, (ast.Name, ast.Attribute))
            for kw in call.keywords
            if kw.arg
        }
        for path, source in current.items():
            if not source or source == pinned.get(path) or path not in pinned:
                continue
            old = dict(_symbols(ast.parse(pinned[path]).body))
            for qualified, func in _symbols(ast.parse(source).body):
                prior = old.get(qualified)
                if not isinstance(func, ast.FunctionDef) or not isinstance(
                    prior, ast.FunctionDef
                ):
                    continue
                for param in (*func.args.args, *func.args.kwonlyargs):
                    name = param.arg
                    if (func.name, name) not in calls or not any(
                        _reads(loop.iter, name)
                        for loop in ast.walk(prior)
                        if isinstance(loop, ast.For)
                    ):
                        continue
                    reads = _reads(func, name)
                    validation = set().union(
                        *(
                            _reads(item.test, name)
                            for item in func.body
                            if isinstance(item, ast.If)
                            and len(item.body) == 1
                            and isinstance(item.body[0], ast.Raise)
                        )
                    )
                    if reads <= validation:
                        return (
                            f"The selected test calls {qualified} with the keyword {name}. On the pinned "
                            f"version, {name} controlled a for loop. The current body reads it only in "
                            "input validation or omits it entirely. Check how this public limit affects "
                            "the new traversal and the observed calls in the fixture."
                        )
    except (SyntaxError, ValueError, RecursionError):
        pass
    return ""
