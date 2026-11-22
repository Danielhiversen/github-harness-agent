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
