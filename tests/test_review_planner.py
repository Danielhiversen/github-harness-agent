from types import SimpleNamespace

from github_harness_agent.providers import ProviderExhausted
from github_harness_agent.review_planner import (
    _SOURCE_LIMIT,
    _advice,
    plan_coding_messages,
)


def test_planner_accepts_one_fenced_json_object():
    inventory = {"demo.py::run": "def run():\n    pass\n"}
    content = '```json\n{"strategy": "Review existing flow.", "symbols": ["demo.py::run"]}\n```'
    assert _advice(content, inventory) == (
        "Review existing flow.",
        "demo.py::run\ndef run():\n    pass\n",
    )
    assert _advice("unrelated prefix\n" + content, inventory) is None


def test_planner_keeps_complete_selected_definitions_within_source_budget():
    first = "def first():\n    " + "pass  # padded\n    " * 220
    omitted = "def omitted():\n    " + "pass  # padded\n    " * 220
    last = "def last():\n    pass\n"
    inventory = {
        "demo.py::first": first,
        "demo.py::omitted": omitted,
        "demo.py::last": last,
    }
    _, sources = _advice(
        '{"strategy": "Review existing flow.", "symbols": ["demo.py::first", "demo.py::omitted", "demo.py::last"]}',
        inventory,
    )
    assert sources == f"demo.py::first\n{first}\ndemo.py::last\n{last}"
    assert len(sources) <= _SOURCE_LIMIT


def test_brief_planner_timeout_preserves_normal_coding_prompt():
    messages = [
        {"role": "system", "content": "Coding task"},
        {
            "role": "user",
            "content": "Selected failure: test_x\nFocused source from demo.py:\ncode\nFailure details:\nfailure",
        },
    ]
    calls = []

    class Budget:
        def complete(self, gateway, role, payload, **options):
            calls.append((gateway, role, payload, options))
            raise ProviderExhausted(None, {"gateway": "request timed out"})

    executor = SimpleNamespace(gateway=object())
    returned = plan_coding_messages(
        executor,
        Budget(),
        messages,
        {"demo.py": "def run():\n    pass\n"},
        "def test_x(): ...",
        "test_x failed",
    )
    assert returned is messages
    assert len(calls) == 1
    assert calls[0][1] == "planning"
    assert calls[0][3]["timeout_cap"] == 25
    assert calls[0][3]["retry_oversize"] is False
