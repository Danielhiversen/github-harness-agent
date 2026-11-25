import json

import pytest

from github_harness_agent.review_definitions import (
    apply_definitions,
    apply_native_definition,
    apply_scoped_raw_definition,
    reviewed_nested_helper,
)
from github_harness_agent.task_context import compact_focused_review


def test_raw_reply_binds_only_to_one_reviewed_nested_helper():
    path = "module.py"
    source = (
        "def run(value):\n"
        "    base = 2\n"
        "    unrelated = 5\n"
        "    def helper(number: int) -> int:\n"
        "        return base + number\n"
        "    return helper(value)\n"
    )
    assert reviewed_nested_helper(source, 2) == (
        "run.helper", "def helper(number: int) -> int:\n    return base + number", ("helper(value)",),
    )
    assert reviewed_nested_helper(source + "\n", 20) is None
    assert reviewed_nested_helper(
        source.replace("    return helper(value)", "    def other():\n        pass\n    return helper(value)"), 2
    ) is None
    symbol = (path, "run.helper")
    replacement = "def helper(number):\n    return base * number"
    assert apply_scoped_raw_definition(replacement, {path: source}, symbol) == {
        path: source.replace("base + number", "base * number"),
    }
    assert apply_scoped_raw_definition(f"```python\n{replacement}\n```", {path: source}, symbol) == {
        path: source.replace("base + number", "base * number"),
    }
    for answer in (
        replacement.replace("def helper", "def other"),
        replacement + "\nvalue = 4",
        "REPLACE module.py::run.helper\n" + replacement,
        f"Here is the helper.\n```python\n{replacement}\n```",
        f"```diff\n{replacement}\n```",
        replacement.replace("def helper(number)", "def helper(number, extra)"),
        "def helper(number):\n    nonlocal undefined_outer_name\n    return base * number",
    ):
        with pytest.raises(ValueError):
            apply_scoped_raw_definition(answer, {path: source}, symbol)
    with pytest.raises(ValueError):
        apply_scoped_raw_definition(replacement, {path: source.replace("def helper", "def extra")}, symbol)


def test_qualified_replace_updates_one_method_and_adds_only_imports():
    original = {
        "module.py": (
            '"""Module docs."""\n\n'
            "class Reader:\n"
            "    def decode(self, raw):\n"
            "        return raw\n\n"
            "    def untouched(self):\n"
            "        return 'unchanged'\n"
        )
    }
    response = """REPLACE module.py::Reader.decode
```python
def decode(self, raw):
    return json.loads(raw)
```
IMPORTS module.py
import json
"""
    result = apply_definitions(response, original)
    assert result["module.py"] == original["module.py"].replace(
        "\n\nclass", "\nimport json\n\nclass"
    ).replace("return raw", "return json.loads(raw)")


@pytest.mark.parametrize(
    "invalid", ["multiple", "duplicate", "extra", "wrong_definition"]
)
def test_native_replacement_rejects_ambiguous_or_unscoped_arguments(invalid):
    code = {"module.py": "def decode(raw):\n    return raw\n"}
    args = {
        "path": "module.py",
        "symbol": "decode",
        "definition": "def decode(raw): return raw.strip()",
    }
    if invalid == "extra":
        args["command"] = "anything"
    if invalid == "wrong_definition":
        args["definition"] = "def other(raw): return raw.strip()"
    serialized = json.dumps(args)
    if invalid == "duplicate":
        serialized = serialized[:-1] + ', "path": "module.py"}'
    call = {
        "type": "function",
        "function": {"name": "replace_python_definition", "arguments": serialized},
    }
    calls = (call, call) if invalid == "multiple" else (call,)
    with pytest.raises(ValueError):
        apply_native_definition(calls, code)
    assert code["module.py"] == "def decode(raw):\n    return raw\n"


def test_bare_fenced_definition_is_bound_to_one_existing_owner():
    code = {"first.py": "def decode(raw):\n    return raw\n"}
    response = "A short edit follows.\n```python\ndef decode(raw):\n    return raw.strip()\n```"
    assert "return raw.strip()" in apply_definitions(response, code)["first.py"]
    with pytest.raises(ValueError, match="ambiguous"):
        apply_definitions(response, {**code, "other.py": code["first.py"]})


@pytest.mark.parametrize("native", (False, True))
def test_replaces_nested_function_by_exact_lexical_owner(native):
    original = (
        "class Reader:\n"
        "    def decode(self, raw):\n"
        "        value = raw\n"
        "        if raw:\n"
        "            @decorate\n"
        "            def normalize(item: str = '') -> str:\n"
        "                nonlocal value\n"
        "                return value + item\n"
        "            return normalize()\n"
        "        return raw\n\n"
        "    def render(self, raw):\n"
        "        def normalize(item):\n"
        "            return item\n"
        "        return normalize(raw)\n"
    )
    body = (
        "@decorate\ndef normalize(item: str = '') -> str:\n"
        "    nonlocal value\n    return value.strip() + item"
    )
    sources = {"module.py": original}

    def replace(symbol, definition):
        if not native:
            return apply_definitions(f"REPLACE module.py::{symbol}\n{definition}", sources)
        call = {
            "type": "function",
            "function": {
                "name": "replace_python_definition",
                "arguments": json.dumps({
                    "path": "module.py", "symbol": symbol, "definition": definition,
                }),
            },
        }
        return apply_native_definition((call,), sources)

    result = replace("Reader.decode.normalize", body)
    assert result["module.py"] == original.replace("return value + item", "return value.strip() + item")
    unannotated = body.replace("item: str = ''", "item = ''").replace(" -> str", "")
    assert replace("Reader.decode.normalize", unannotated) == result
    for symbol in ("normalize", "decode.normalize", "Reader.normalize"):
        with pytest.raises(ValueError):
            replace(symbol, body)
    with pytest.raises(ValueError, match="signature and decorators"):
        replace("Reader.decode.normalize", body.replace("item: str = ''", "item"))
    with pytest.raises(ValueError, match="signature and decorators"):
        replace("Reader.decode.normalize", body.removeprefix("@decorate\n"))
    with pytest.raises(ValueError, match="signature and decorators"):
        replace("Reader.decode.normalize", body.replace("-> str", "-> bytes"))
    with pytest.raises(ValueError, match="signature and decorators"):
        replace("Reader.render.normalize", "def normalize(item: str):\n    return item")
    with pytest.raises(ValueError, match="syntax|valid"):
        replace("Reader.decode.normalize", body.replace("nonlocal value", "nonlocal missing"))


def test_omitted_annotations_splice_body_under_exact_multiline_declaration():
    source = (
        "def outer():\n"
        "    @decorate ('value')\n"
        "    def inner(\n"
        "        α: str, /, *args: str, flag: bool = False, **kwargs: str\n"
        "    ) -> str:  # preserve exact header\n"
        "        # retain the declaration's leading comment\n"
        "        return α\n"
        "    return inner('input')\n"
    ).replace("\n", "\r\n")
    reply = (
        "REPLACE module.py::outer.inner\n"
        "@decorate('value')\n"
        "def inner(α, /, *args, flag=False, **kwargs): return α.upper()\n"
    )
    result = apply_definitions(reply, {"module.py": source})
    assert result["module.py"] == source.replace("return α\r\n", "return α.upper()\r\n")
    signature = "α, /, *args, flag=False, **kwargs"
    for changed in (
        "α, *args, flag=False, **kwargs",
        "α, /, *inputs, flag=False, **kwargs",
        "α, /, *args, flag=True, **kwargs",
    ):
        with pytest.raises(ValueError, match="signature and decorators"):
            apply_definitions(reply.replace(signature, changed), {"module.py": source})


@pytest.mark.parametrize("invalid", (
    "def test_keep():\n    assert True",
    "def test_failed(other):\n    assert other",
    "@pytest.mark.skip\ndef test_failed(self):\n    assert True",
    "import pytest\ndef test_failed(self):\n    assert True",
    "def test_failed(self):\n    assert True\n# extra",
))
def test_raw_collected_test_repair_rejects_siblings_or_contract_change(invalid):
    from github_harness_agent.review_definitions import apply_scoped_raw_definition

    source = (
        "class TestCases:\n"
        "    def test_failed(self):\n        assert False\n\n"
        "    def test_keep(self):\n        assert True\n"
    )
    sources = {"tests/test_draft.py": source}
    symbol = ("tests/test_draft.py", "TestCases.test_failed")
    with pytest.raises(ValueError):
        apply_scoped_raw_definition(invalid, sources, symbol, require_nested=False)
    assert sources[symbol[0]] == source


def test_omitted_annotations_preserve_inline_declaration_and_replace_suite():
    source = "def outer():\n    def inner(value: str): return value\n    return inner('input')\n"
    reply = "REPLACE module.py::outer.inner\ndef inner(value):\n    upper = value.upper()\n    return upper"
    result = apply_definitions(reply, {"module.py": source})
    assert result["module.py"] == source.replace(
        "def inner(value: str): return value",
        "def inner(value: str):\n        upper = value.upper()\n        return upper",
    )


def test_nested_function_rejects_duplicate_owners_and_overlapping_edits():
    before = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    inner = "REPLACE module.py::outer.inner\ndef inner():\n    return 2\n"
    with pytest.raises(ValueError, match="ambiguous"):
        apply_definitions(inner, {"module.py": before + "\ndef outer():\n    return 0\n"})
    with pytest.raises(ValueError, match="overlap"):
        apply_definitions(inner + "REPLACE module.py::outer\n" + before, {"module.py": before})
    result = apply_definitions(inner, {"module.py": before})
    assert result["module.py"] == before.replace("return 1", "return 2")


def test_class_and_nested_method_replacements_cannot_overlap():
    code = {"module.py": "class Reader:\n    def read(self):\n        return 1\n"}
    response = (
        "REPLACE module.py::Reader\nclass Reader:\n    def read(self):\n        return 2\n"
        "REPLACE module.py::Reader.read\ndef read(self):\n    return 3\n"
    )
    with pytest.raises(ValueError, match="overlap"):
        apply_definitions(response, code)


def test_named_definition_prompt_remains_eligible_for_compact_gateway_retry():
    footer = "Respond with complete named Python definitions only."
    messages = [
        {"role": "system", "content": "Named replacements."},
        {
            "role": "user",
            "content": "Selected failure: test_decode\nReviewer feedback: keep encoding\n"
            "Focused source from reader.py:\ndef decode(raw): return raw\n"
            "\nCurrent checkout:\n" + "bulk\n" * 4000 + footer,
        },
    ]
    compact = compact_focused_review(messages, max_bytes=12 * 1024)
    assert "keep encoding" in compact[1]["content"]
    assert "def decode" in compact[1]["content"]
    assert "bulk" not in compact[1]["content"]
    assert compact[1]["content"].endswith(footer)


@pytest.mark.parametrize(
    "bad_block",
    [
        "REPLACE ../other.py::decode\ndef decode(raw):\n    return raw",
        "REPLACE module.py::decode\ndef renamed(raw):\n    return raw",
        "REPLACE module.py::decode\nasync def decode(raw):\n    return raw",
        "IMPORTS module.py\nflag = True",
        "IMPORTS module.py\nfrom __future__ import annotations",
    ],
)
def test_invalid_block_rejects_the_whole_batch(bad_block):
    code = {"module.py": "def decode(raw):\n    return raw\n"}
    response = (
        "REPLACE module.py::decode\ndef decode(raw):\n    return raw.strip()\n"
        + bad_block
    )
    with pytest.raises(ValueError):
        apply_definitions(response, code)
    assert code["module.py"] == "def decode(raw):\n    return raw\n"
