from github_harness_agent.review_planner import (
    _localized_definition,
    _StockAdvice,
    restore_pinned_stock_definition,
)


def test_pinned_restore_replaces_only_the_exact_bound_method():
    path = "feature.py"
    key = f"{path}::Example.calculate"
    before = (
        "class Example:\n"
        "    def calculate(self):\n"
        + "        # preserve the complete pinned body\n"
        * 150
        + "        return 1\n\n"
        "    def other(self):\n"
        "        return 3\n"
    )
    original = {path: before, "peer.py": "value = 3\n"}
    current = {
        path: before.replace("return 1", "return 2").replace("return 3", "return 4"),
        "peer.py": "value = 8\n",
    }
    localized = _localized_definition(original, current, key, 5)
    advice = _StockAdvice("repair", "", "", localized, "")
    restored = restore_pinned_stock_definition(
        advice, original, current, {path}, minimum_bytes=6100
    )
    assert restored == {
        path: current[path].replace("return 2", "return 1"),
        "peer.py": "value = 8\n",
    }
    assert current[path] != restored[path]
    stale = {**current, path: current[path].replace("return 4", "return 5")}
    assert (
        restore_pinned_stock_definition(
            advice, original, stale, {path}, minimum_bytes=1
        )
        is None
    )
    assert (
        restore_pinned_stock_definition(
            advice, original, current, set(), minimum_bytes=1
        )
        is None
    )
    ambiguous = {**current, path: current[path] + "\nclass Example: pass\n"}
    ambiguous_advice = _StockAdvice(
        "repair", "", "", _localized_definition(original, ambiguous, key, 5), ""
    )
    assert (
        restore_pinned_stock_definition(
            ambiguous_advice, original, ambiguous, {path}, minimum_bytes=1
        )
        is None
    )


_PINNED_DEPENDENCIES = """from adapters import decode, unused
from io import StringIO

class Example:
    def calculate(self, value):
        return decode(self.prepare(StringIO(value)))

    def prepare(self, value) -> list[str]:
        return [value.read()]

    def sibling(self):
        return 7
"""
_DRAFT_DEPENDENCIES = (
    _PINNED_DEPENDENCIES.replace(
        "from adapters import decode, unused\n", "unused = 'draft'\n"
    )
    .replace("from io import StringIO\n", "")
    .replace("decode(self.prepare(StringIO(value)))", "self.prepare(value)")
    .replace("-> list[str]", "-> dict[str, str]")
    .replace("return [value.read()]", "return {'draft': value}")
    .replace("return 7", "return 8")
)


def _prepare_restore(before, current):
    from github_harness_agent.review_planner import prepare_pinned_stock_restore

    path = "feature.py"
    original, draft = {path: before}, {path: current}
    row = next(
        i
        for i, line in enumerate(current.splitlines(), start=1)
        if "def calculate(" in line
    )
    localized = _localized_definition(
        original, draft, f"{path}::Example.calculate", row
    )
    advice = _StockAdvice("repair", "", "", localized, "")
    sources, summary = prepare_pinned_stock_restore(
        advice, original, draft, {path}, minimum_bytes=1
    )
    return sources[path], summary


def test_pinned_restore_includes_one_container_change_and_only_missing_import_bindings():
    restored, summary = _prepare_restore(_PINNED_DEPENDENCIES, _DRAFT_DEPENDENCIES)
    assert "from adapters import decode\n" in restored
    assert "from adapters import decode, unused" not in restored
    assert "from io import StringIO\n" in restored
    assert "unused = 'draft'" in restored
    assert "return decode(self.prepare(StringIO(value)))" in restored
    assert "def prepare(self, value) -> list[str]" in restored
    assert "return [value.read()]" in restored
    assert "def sibling(self):\n        return 8" in restored
    assert "list" in summary and "dict" in summary and len(summary.encode()) <= 480


def test_pinned_restore_does_not_choose_between_two_changed_direct_helpers():
    before = (
        _PINNED_DEPENDENCIES.replace(
            "return decode(self.prepare(StringIO(value)))",
            "return decode(self.prepare(StringIO(value))) + self.combine(value)",
        )
        + "\n    def combine(self, value) -> list[str]:\n        return [value]\n"
    )
    current = (
        _DRAFT_DEPENDENCIES
        + "\n    def combine(self, value) -> dict[str, str]:\n        return {'draft': value}\n"
    )
    restored, summary = _prepare_restore(before, current)
    assert "def prepare(self, value) -> dict[str, str]" in restored
    assert "def combine(self, value) -> dict[str, str]" in restored
    assert "ambiguous" in summary


def test_pinned_restore_keeps_existing_conflicting_and_conditional_import_bindings():
    draft = (
        "from replacement import decode\n"
        "if condition:\n"
        "    from replacement import StringIO\n" + _DRAFT_DEPENDENCIES
    )
    restored, summary = _prepare_restore(_PINNED_DEPENDENCIES, draft)
    assert "from replacement import decode" in restored
    assert "    from replacement import StringIO" in restored
    assert "from adapters import decode" not in restored
    assert "from io import StringIO" not in restored
    assert "decode" in summary and "StringIO" in summary


def test_pinned_dependency_restore_does_not_rebind_names_used_by_other_draft_code():
    current = _DRAFT_DEPENDENCIES.replace("return 8", "return decode('unrelated')")
    restored, summary = _prepare_restore(_PINNED_DEPENDENCIES, current)
    assert "from adapters import decode" not in restored
    assert "from io import StringIO\n" in restored
    assert "return decode('unrelated')" in restored
    assert "shared draft bindings" in summary


def test_pinned_dependency_advisory_survives_compact_observations():
    from github_harness_agent.review_mock_replay import compact_observations

    _, summary = _prepare_restore(_PINNED_DEPENDENCIES, _DRAFT_DEPENDENCIES)
    text = (
        "Auxiliary counterfactual (quoted data). "
        + "older evidence " * 200
        + "\n"
        + summary
    )
    compact = compact_observations(text, 1080)
    assert summary in compact and len(compact.encode()) <= 1080


def test_pinned_restore_never_selects_arbitrarily_from_more_than_two_imports():
    before = ("from decimal import Decimal\n" + _PINNED_DEPENDENCIES).replace(
        "decode(self.prepare(StringIO(value)))",
        "decode(self.prepare(StringIO(value)), Decimal(1))",
    )
    restored, summary = _prepare_restore(before, _DRAFT_DEPENDENCIES)
    assert "from decimal import Decimal" not in restored
    assert "from adapters import decode" not in restored
    assert "from io import StringIO" not in restored
    assert "More than two absent import bindings" in summary


def test_pinned_seed_marks_verified_dependency_edits_only():
    from github_harness_agent.review_planner import pinned_stock_restore_seed

    path = "feature.py"
    original = {path: _PINNED_DEPENDENCIES}
    with_dependencies = {path: _DRAFT_DEPENDENCIES}
    primary_only = {
        path: _PINNED_DEPENDENCIES.replace(
            "return decode(self.prepare(StringIO(value)))", "return self.prepare(value)"
        )
    }
    for draft, has_dependency_edits in (
        (with_dependencies, True),
        (primary_only, False),
    ):
        row = next(
            i
            for i, line in enumerate(draft[path].splitlines(), 1)
            if "def calculate(" in line
        )
        localized = _localized_definition(
            original, draft, f"{path}::Example.calculate", row
        )
        advice = _StockAdvice("repair", "", "", localized, "")
        seed = pinned_stock_restore_seed(
            advice, original, draft, {path}, minimum_bytes=1
        )
        assert seed.has_dependency_edits is has_dependency_edits
