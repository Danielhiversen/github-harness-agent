import hashlib
from datetime import UTC, datetime

import pytest

from github_harness_agent.review_mock_replay import (
    CounterCallerLocation,
    CounterLoopReuse,
)
from github_harness_agent.review_semantic_key import guard_traced_counter_once

PATH = "service.py"
SOURCE = """from collections import deque
from datetime import UTC, datetime

def count(rows, deny):
    counters = {}
    candidates = {}
    def report(reason):
        counters[reason] = counters.get(reason, 0) + 1

    for entry in rows:
        if deny(entry):
            report("excluded")
            continue
        candidates.setdefault("north", deque()).append(entry)

    for group, bucket in candidates.items():
        candidates[group] = deque(sorted(
            bucket,
            key=lambda item: (
                str(item.get("label")),
                str(item["observed"]),
                item["x"],
                item["y"],
            ),
        ))
    return counters, candidates
"""


def observed(source=SOURCE, *, pinned=1):
    def row(text):
        return source.splitlines().index(text) + 1

    def location(name, text):
        return CounterCallerLocation(
            PATH, name, row(text), hashlib.sha256(source.encode()).hexdigest()
        )

    return CounterLoopReuse(
        location("count", '            report("excluded")'),
        location(
            "count.report", "        counters[reason] = counters.get(reason, 0) + 1"
        ),
        row("        if deny(entry):"),
        "entry",
        2,
        2,
        pinned,
        "tests/test_counts.py::test_counter",
        5,
    )


def invoke(source, entries, deny=lambda entry: True):
    namespace = {}
    exec(compile(source, PATH, "exec"), namespace)
    return namespace["count"](entries, deny)


def test_proved_group_sort_key_guards_only_the_traced_effect():
    start = datetime(2025, 4, 8, tzinfo=UTC)
    base = {"label": "a", "observed": start, "x": 9, "y": 3.0}
    same_key = {**base, "unrelated": 3}
    another_key = {**base, "observed": start.replace(day=9)}
    another_position = {**base, "x": 10}
    direct = guard_traced_counter_once({PATH: SOURCE}, observed(), {PATH})
    assert direct
    assert len(direct[PATH].splitlines()) - len(SOURCE.splitlines()) <= 23
    assert "_gha_" not in direct[PATH]
    assert "            continue\n" in direct[PATH]
    assert invoke(SOURCE, [base, same_key])[0] == {"excluded": 2}
    assert invoke(direct[PATH], [base, same_key])[0] == {"excluded": 1}
    assert invoke(direct[PATH], [base, another_key, another_position])[0] == {
        "excluded": 3
    }
    assert invoke(direct[PATH], [base, same_key], lambda entry: False) == invoke(
        SOURCE, [base, same_key], lambda entry: False
    )
    assert invoke(direct[PATH], [base, same_key])[0] == {"excluded": 1}

    class Unexpected:
        def __str__(self):
            raise AssertionError("new conversion must not execute")

    incomplete = {"label": Unexpected()}
    unusual = {**base, "label": Unexpected()}
    assert invoke(direct[PATH], [incomplete, incomplete, unusual, unusual])[0] == {
        "excluded": 4
    }


@pytest.mark.parametrize(
    "case",
    [
        "spoof",
        "stale",
        "protected",
        "incomparable",
        "impure",
        "mixed",
        "unrelated",
        "ambiguous",
        "rebound",
        "shadowed builtin",
        "shadowed deque",
    ],
)
def test_semantic_key_trial_declines_unproven_sources(case):
    source = SOURCE
    if case == "impure":
        source = source.replace('item["x"]', 'compute(item["x"])')
    elif case == "mixed":
        source = source.replace(
            "    for group, bucket",
            '    candidates.setdefault("north", deque()).append({})\n    for group, bucket',
        )
    elif case == "unrelated":
        source = source.replace("in candidates.items()", "in unrelated.items()")
    elif case == "rebound":
        source = source.replace(
            "        candidates.setdefault(",
            "        entry = {}\n        candidates.setdefault(",
        )
    elif case == "shadowed builtin":
        source = "str = custom_string\n" + source
    elif case == "shadowed deque":
        source += "\ndeque = CustomDeque\n"
    elif case == "ambiguous":
        source = source.replace(
            "    return counters, candidates",
            '    sorted(rows, key=lambda item: (item["x"], item["y"]))\n    return counters, candidates',
        )
    signal = observed(
        SOURCE if case == "stale" else source, pinned=2 if case == "incomparable" else 1
    )
    if case == "stale":
        source += "\n"
    if case == "spoof":
        signal = str(signal)
    assert (
        guard_traced_counter_once(
            {PATH: source}, signal, set() if case == "protected" else {PATH}
        )
        is None
    )
