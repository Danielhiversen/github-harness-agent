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
    _, _, advice = _coding_completion(
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
    assert calls[1][2]["max_output_tokens"] == 3072
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
