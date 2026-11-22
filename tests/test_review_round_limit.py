import pytest

from github_harness_agent.review_round_limit import prepare_round_limit_guard

PATH = "scheduler.py"
OPERATOR = """def test_depth():
    assert Scheduler().run(rounds=1) == [2]
    assert sorted(Scheduler().run(rounds=2)) == [2, 3, 4]
"""
SHARED = """from __future__ import annotations
from collections import deque

class Work:
    def __init__(self, value, generation=0):
        self.value = value
        self.generation = generation

    def children(self) -> list[Work]:
        \"\"\"Produce the next round's items.\"\"\"
        if self.value >= 4:
            return []
        return [Work(value, self.generation + 1) for value in range(self.value + 1, self.value + 3)]

class Scheduler:
"""
PINNED = (
    SHARED
    + """    def seed(self):
        roots = []
        roots.append(Work(2))
        return roots

    def run(self, *, rounds=5):
        if min(rounds, 100) < 1:
            raise ValueError("limits")
        nodes = self.seed()
        result = []
        for _ in range(rounds):
            following = []
            for item in nodes:
                result.append(item.value)
                if item.value < 5:
                    following.extend(item.children())
            nodes = following
        return result
"""
)
QUEUED = (
    SHARED
    + """    def seed(self):
        roots = {}
        roots.setdefault("one", deque()).append(Work(2))
        return roots

    def run(self, *, rounds=5):
        if min(rounds, 100) < 1:
            raise ValueError("limits")
        roots = self.seed()
        deferred = {}
        labels = deque(roots)
        result = []
        while labels:
            label = labels.popleft()
            if deferred.get(label):
                item = deferred[label].popleft()
            elif roots[label]:
                item = roots[label].popleft()
            else:
                continue
            result.append(item.value)
            if item.value < 5:
                children = item.children()
                for child in children:
                    deferred.setdefault(label, deque()).append(child)
            if roots[label] or deferred.get(label):
                labels.append(label)
        return result
"""
)


def prepare(pinned=PINNED, current=QUEUED, fixture=OPERATOR, allowed=(PATH,)):
    return prepare_round_limit_guard(
        {PATH: pinned},
        {PATH: current, "unrelated.py": "UNCHANGED = True\n"},
        allowed,
        owner=(PATH, "Scheduler.run"),
        operator_fixture=fixture,
    )


def run(source, rounds):
    module = {}
    exec(compile(source, PATH, "exec"), module)
    return sorted(module["Scheduler"]().run(rounds=rounds))


def test_round_guard_recovers_pinned_rounds_and_only_changes_factory_expression():
    assert run(QUEUED, 1) != run(PINNED, 1)
    assert run(QUEUED, 2) != run(PINNED, 2)
    proposal = prepare()
    assert proposal is not None
    assert proposal.keyword == "rounds" and proposal.factory == "Work.children"
    assert "range(rounds)" in proposal.advisory
    assert "self.generation = generation" in proposal.advisory
    assert "self.generation + 1" in proposal.advisory
    assert "item.value < 5" in proposal.advisory
    assert "Work(2)" in proposal.advisory
    assert proposal.sources["unrelated.py"] == "UNCHANGED = True\n"
    edited = proposal.sources[PATH]
    assert (
        edited.replace(
            "(item.children() if item.generation + 1 < rounds else [])",
            "item.children()",
        )
        == QUEUED
    )
    for rounds in range(1, 5):
        assert run(edited, rounds) == run(PINNED, rounds)


@pytest.mark.parametrize(
    "case",
    [
        "no fixture",
        "unreviewed",
        "no pinned round",
        "still in use",
        "nonzero default",
        "different root",
        "nonzero root",
        "mixed increments",
        "removed factory",
        "second factory",
        "ambiguous expansion",
        "no enqueue",
        "no matching pop",
    ],
)
def test_round_limit_proof_declines_near_misses(case):
    pinned, current, fixture, allowed = PINNED, QUEUED, OPERATOR, (PATH,)
    if case == "no fixture":
        fixture = "def test_other():\n    assert Other().unrelated(args=1)"
    elif case == "unreviewed":
        allowed = ()
    elif case == "no pinned round":
        pinned = pinned.replace("in range(rounds)", "in range(5)")
    elif case == "still in use":
        current = current.replace("while labels:", "while labels and rounds:")
    elif case == "nonzero default":
        pinned, current = (
            text.replace("generation=0", "generation=1") for text in (pinned, current)
        )
    elif case == "different root":
        current = current.replace("def seed(self)", "def different(self)").replace(
            "self.seed()", "self.different()"
        )
    elif case == "nonzero root":
        current = current.replace("append(Work(2))", "append(Work(2, 1))")
    elif case == "mixed increments":
        current = current.replace("self.generation + 1", "self.generation + 2")
    elif case == "removed factory":
        current = current.replace("def children(", "def removed(")
    elif case == "second factory":
        both = [
            text.replace(
                "class Scheduler:",
                "    def alternate(self):\n        return Work(self.value)\n\nclass Scheduler:",
            )
            for text in (pinned, current)
        ]
        pinned, current = both
    elif case == "ambiguous expansion":
        current = current.replace(
            "            if roots[label] or",
            "            if item.value < 5:\n                children = item.children()\n                deferred[label].extend(children)\n            if roots[label] or",
        )
    elif case == "no enqueue":
        current = current.replace(
            "deferred.setdefault(label, deque()).append(child)", "result.append(child)"
        )
    elif case == "no matching pop":
        current = current.replace("item = roots[label].popleft()", "item = Work(2)")
    assert prepare(pinned, current, fixture, allowed) is None


def indirect_queue():
    return QUEUED.replace(
        "        deferred = {}", "        deferred = {key: deque() for key in roots}"
    ).replace(
        "                for child in children:\n"
        "                    deferred.setdefault(label, deque()).append(child)",
        "                groups = {}\n"
        "                for child in children:\n"
        "                    groups.setdefault(label, []).append(child)\n"
        "                for group in sorted(groups.keys()):\n"
        "                    batch = groups[group]\n"
        "                    deferred.setdefault(group, deque()).extendleft(reversed(batch))",
    )


def test_round_guard_proves_child_only_regrouping_and_preserves_it():
    source = indirect_queue()
    proposal = prepare(current=source)
    assert proposal is not None
    assert run(proposal.sources[PATH], 1) == run(PINNED, 1)
    assert run(proposal.sources[PATH], 2) == run(PINNED, 2)
    assert ".extendleft(reversed(batch))" in proposal.sources[PATH]


@pytest.mark.parametrize(
    "sequence",
    [
        "                groups.setdefault(label, []).append(Work(9))",
        "                deferred.setdefault(label, deque()).append(Work(9))",
        "                deferred[label] = deque([Work(9)])",
    ],
)
def test_round_guard_declines_foreign_worklist_insertions(sequence):
    source = indirect_queue().replace(
        "                for group in sorted",
        sequence + "\n                for group in sorted",
    )
    assert prepare(current=source) is None


def test_round_guard_uses_literal_loop_target_values_from_operator_fixture():
    fixture = """def test_contract():
    for steps, expected in ((1, {2}), (2, {2, 3, 4})):
        assert set(Scheduler().run(rounds=steps)) == expected
"""
    proposal = prepare(fixture=fixture)
    assert proposal is not None
    assert prepare(fixture=fixture.replace("(1, {2})", "(0, {2})")) is None
    assert (
        prepare(
            fixture=fixture.replace(
                "        assert", "        steps = unknown\n        assert"
            )
        )
        is None
    )
