from types import SimpleNamespace

import pytest

from github_harness_agent.review_definitions import DEFINITION_ENDING, apply_definitions
from github_harness_agent.review_planner import localized_retry_symbol
from github_harness_agent.review_solver import (
    _DEFINITION_SYSTEM,
    _SYSTEM,
    _coding_completion,
)


PATH = "pkg/queue.py"
KEY = f"{PATH}::Queue.receive"
STOCK = "def test_tracked(queue):\n    assert queue.receive(1) == 4\n"
OPERATOR = "def test_verified(queue):\n    assert queue.receive(5) == 8\n"


def _messages(named):
    ending = (
        DEFINITION_ENDING if named else "Respond with exact SEARCH/REPLACE blocks only."
    )
    return [
        {"role": "system", "content": _DEFINITION_SYSTEM if named else _SYSTEM},
        {
            "role": "user",
            "content": "Selected failure: tests/test_queue.py::test_tracked\n"
            f"Editable checkout paths (use these exact names): {PATH}\n"
            "Previously rejected edit: HISTORIC_REJECTED_HUNK\n"
            f"Focused source from {PATH}:\n# abbreviated\n"
            "Failure details:\nstock failure\n"
            "Last proposal feedback for this goal: HISTORIC_REJECTED_HUNK\n"
            f"\nFixture for this goal:\n{STOCK}\n"
            "\nCurrent checkout:\n" + ending,
        },
    ]


@pytest.mark.parametrize(
    ("line", "padding", "localized"),
    [(5, 120, True), (1, 120, False), (5, 700, False)],
)
def test_stock_named_retry_keeps_exact_pair_and_existing_strategy(
    line, padding, localized
):
    old = (
        "    def receive(self, value):\n"
        "        sent = 3\n"
        + "        # pinned behavior\n" * padding
        + "        return value + sent\n"
    )
    current = old.replace("sent = 3", "sent = 5")
    original = {PATH: "class Queue:\n" + old}
    sources = {PATH: "class Queue:\n" + current}
    calls = []

    def complete(gateway, role, messages, *rest, **options):
        calls.append((role, messages, options))
        return SimpleNamespace(
            content='{"strategy": "Trace the public inputs.", "symbols": ["pkg.queue.Queue.receive"]}'
            if role == "planning"
            else "a coding proposal"
        )

    executor = SimpleNamespace(gateway=object())
    budget = SimpleNamespace(complete=complete)
    mock_detail = (
        "Auxiliary counterfactual (quoted data, partial diagnostic only). "
        f"{KEY}, draft line {line}: if value."
    )
    _, _, advice, _ = _coding_completion(
        executor,
        budget,
        _messages(False),
        None,
        (
            sources,
            STOCK,
            "stock failure",
            mock_detail,
            (original, {"tests/test_operator.py": OPERATOR}, []),
        ),
    )
    retry_symbol = localized_retry_symbol(advice)
    assert retry_symbol == ((PATH, "Queue.receive") if localized else None)
    _coding_completion(
        executor, budget, _messages(localized), None, None, advice=advice
    )
    assert [role for role, _, _ in calls] == ["planning", "coding", "coding"]
    fresh, retry = calls[1][1], calls[2][1]
    assert "Trace the public inputs." in fresh[-1]["content"]
    assert fresh[-1]["content"].endswith(
        "Respond with exact SEARCH/REPLACE blocks only."
    )
    assert old not in fresh[-1]["content"]
    assert calls[1][2]["max_output_tokens"] == (6144 if localized else 3072)
    if not localized:
        assert retry[-1]["content"].endswith(
            "Respond with exact SEARCH/REPLACE blocks only."
        )
        return
    prompt = retry[-1]["content"]
    assert "Trace the public inputs." in prompt
    assert f"REPLACE {KEY}" in prompt
    assert old in prompt and current in prompt
    assert "read-only" in prompt
    assert "assert queue.receive(1) == 4" in prompt
    assert "assert queue.receive(5) == 8" in prompt
    assert "HISTORIC_REJECTED_HUNK" not in prompt
    assert "<<<<<< SEARCH" not in retry[0]["content"]
    assert prompt.endswith(DEFINITION_ENDING)
    assert calls[2][2]["max_output_tokens"] == 6144


def test_named_retry_rejects_edits_outside_localized_definition():
    sources = {
        PATH: "class Queue:\n    def receive(self):\n        return 1\ndef helper():\n    return 2\n"
    }
    accepted = f"REPLACE {KEY}\ndef receive(self):\n    return 3"
    assert (
        "return 3"
        in apply_definitions(accepted, sources, only=(PATH, "Queue.receive"))[PATH]
    )
    with pytest.raises(ValueError, match="localized"):
        apply_definitions(
            accepted + f"\nREPLACE {PATH}::helper\ndef helper():\n    return 4",
            sources,
            only=(PATH, "Queue.receive"),
        )


def test_verified_counter_caller_localizes_and_survives_constrained_sparse_prompt():
    from github_harness_agent.review_line_ranges import prepare_line_range_edit
    from github_harness_agent.review_mock_replay import (
        _counter_difference,
        counter_caller_site,
    )
    from github_harness_agent.review_planner import request_coding_advice
    import httpx
    from test_providers import GROQ, deployment, openai_response, routed

    from github_harness_agent.task_context import compact_focused_review, wire_size

    old = (
        "    def receive(self, value):\n"
        "        if value > 4:\n"
        "            self.fallback(value)\n"
        + "        # Keep the exact localized definition.\n" * 140
        + "        return value + 3\n"
    )
    current = old.replace("value > 4", "value >= 4").replace(
        "self.fallback(value)", "self.increment(value)"
    )
    prefix = (
        "class Queue:\n"
        "    def fallback(self, value):\n"
        "        if value < 2:\n"
        "            self.increment(value)\n\n"
    )
    original = {PATH: prefix + old}
    sources = {PATH: prefix + current}
    shared = [PATH, "Queue.increment", 1]
    pinned_stack = [shared, [PATH, "Queue.fallback", 4], [PATH, "Queue.receive", 8]]
    draft_stack = [shared, [PATH, "Queue.receive", 8]]
    report = _counter_difference(
        [1, None, [["a" * 64, [[pinned_stack, 0, 1]]]]],
        [1, 0, [["a" * 64, [[draft_stack, 0, 1]]]]],
        "test_queue.py::test_tracked",
        (original, sources),
    )
    calls = []

    def complete(gateway, role, messages, *rest, **options):
        calls.append((role, messages, options))
        return SimpleNamespace(
            content='{"strategy":"Trace the correct dispatch.","symbols":["pkg.queue.Queue.receive"]}'
        )

    executor = SimpleNamespace(gateway=object())
    budget = SimpleNamespace(complete=complete)
    planner_args = (
        executor,
        budget,
        _messages(False),
        sources,
        STOCK,
        "stock failure",
        str(report),
        (original, {"tests/test_operator.py": OPERATOR}, []),
    )
    advice = request_coding_advice(
        *planner_args, counter_site=counter_caller_site(report)
    )
    assert calls[0][2]["timeout_cap"] == 120
    assert current in calls[0][1][-1]["content"]
    assert localized_retry_symbol(advice) == (PATH, "Queue.receive")
    retry = _messages(True)
    retry[-1]["content"] = retry[-1]["content"].replace(
        "stock failure", "stock failure\n" + str(report)
    )
    _coding_completion(executor, budget, retry, None, None, advice=advice)
    assert old in calls[1][1][-1]["content"] and current in calls[1][1][-1]["content"]
    line_snapshot = prepare_line_range_edit(sources, (PATH, "Queue.receive"))
    _coding_completion(
        executor, budget, retry, None, None, advice=advice, line_snapshot=line_snapshot
    )
    requests = []

    def handle(request):
        import json

        messages = json.loads(request.content)["messages"]
        requests.append(messages)
        return httpx.Response(200, json=openai_response())

    assert wire_size(calls[2][1]) > 12 * 1024
    router = routed(deployment(GROQ), handle)
    router.complete(role="coding", messages=calls[2][1], max_output_tokens=3072)
    assert len(requests) == 1 and wire_size(requests[0]) <= 12 * 1024
    prompt = requests[0][-1]["content"]
    assert line_snapshot.digest in prompt and "return value + 3" in prompt
    assert "Pinned caller source" in prompt and "Draft caller source" in prompt
    assert "Pinned outer caller source" in prompt
    assert "self.fallback(value)" in prompt and "if value > 4:" in prompt
    assert "if value < 2:" in prompt and "if value >= 4:" in prompt
    assert "assert queue.receive(5) == 8" in prompt
    # An even tighter provider keeps the immediate pair ahead of the extra window.
    tighter = [
        candidate
        for cap in range(8000, 12 * 1024, 32)
        if (candidate := compact_focused_review(calls[2][1], max_bytes=cap))
        and "Pinned caller source" in candidate[-1]["content"]
        and "Pinned outer caller source" not in candidate[-1]["content"]
    ]
    assert tighter and all(
        "Draft caller source" in item[-1]["content"] for item in tighter
    )
    untrusted = request_coding_advice(*planner_args)
    assert localized_retry_symbol(untrusted) is None
    assert calls[-1][2]["timeout_cap"] == 60

    stale = list(planner_args)
    stale[3] = {PATH: sources[PATH].replace("value >= 4", "value >= 5")}
    rejected = request_coding_advice(*stale, counter_site=counter_caller_site(report))
    assert localized_retry_symbol(rejected) is None
    assert calls[-1][2]["timeout_cap"] == 60
