import json

import pytest

from github_harness_agent.review_definitions import (
    apply_definitions,
    apply_native_definition,
)
from github_harness_agent.task_context import compact_focused_review


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
