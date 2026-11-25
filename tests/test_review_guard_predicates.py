import pytest

from github_harness_agent.review_guards import guarded_sources


@pytest.mark.parametrize(
    ("parameter", "setup", "selector", "selected", "other"),
    [
        ("enabled: bool", "", "enabled", True, False),
        ("tag: str | None", "", "tag is None", None, "tag"),
        (
            "mode: str",
            '    if mode == "strict":\n        status = 0\n',
            "mode == 'strict'",
            "strict",
            "loose",
        ),
    ],
)
def test_guarded_predicate_preserves_exact_branch_source(
    parameter, setup, selector, selected, other
):
    body = (
        "        # Preserve this comment and every statement.\n"
        + "        status += 1\n" * 70
        + "    else:\n        status = -1\n"
        "    return status\n"
    )
    before = (
        f"def check(value, {parameter}):\n"
        "    status = 0\n"
        + setup
        + "    if value >= 3:  # preserve the inline comment\n"
        + body
    )
    after = before.replace("value >= 3", "value >= 5")
    candidates = list(guarded_sources(before, after))
    assert len(candidates) == 2
    for candidate, expected in zip(candidates, [(-1, 70), (70, -1)], strict=True):
        assert (
            candidate[: candidate.index("    if (")]
            == before[: before.index("    if value")]
        )
        assert candidate.endswith("  # preserve the inline comment\n" + body)
        assert candidate.count(body) == 1
        assert f"if ({selector}) else" in candidate
        scope = {}
        exec(candidate, scope)
        assert (scope["check"](4, selected), scope["check"](4, other)) == expected


def test_guarded_predicate_rejects_ambiguous_selector_and_unrelated_source():
    body = "        score += 1\n" * 16 + "    return score\n"
    normal = "def check(value, enabled: bool):\n    score = 0\n    if value:\n" + body
    changed = normal.replace("if value:", "if value > 0:")
    assert (
        list(guarded_sources(normal, changed.replace("return score", "return 9"))) == []
    )
    assert (
        list(
            guarded_sources(
                normal, changed.replace("if value > 0:", "if value > 0:  # new")
            )
        )
        == []
    )
    ambiguous = normal.replace("enabled: bool", "enabled: bool, verbose: bool")
    assert (
        list(
            guarded_sources(ambiguous, ambiguous.replace("if value:", "if value > 0:"))
        )
        == []
    )
    shadowed = normal.replace(
        "    score = 0\n", "    import helper as enabled\n    score = 0\n"
    )
    assert (
        list(guarded_sources(shadowed, shadowed.replace("if value:", "if value > 0:")))
        == []
    )


def test_guarded_predicate_supports_multiline_expression_with_unicode():
    before = """def check(values, enabled: bool):
    hits = []
    for café in values:
        if (
            café >= 1
            and café <= 3
        ):  # keep the header comment
            hits.append(café)
    return hits
"""
    after = before.replace("café >= 1", "café >= 2").replace("café <= 3", "café <= 4")
    forward, reverse = guarded_sources(before, after)
    for candidate, expected in (
        (forward, ([2, 3, 4], [1, 2, 3])),
        (reverse, ([1, 2, 3], [2, 3, 4])),
    ):
        assert candidate.count("# keep the header comment") == 1
        assert candidate.count("hits.append(café)") == 1
        scope = {}
        exec(candidate, scope)
        assert (
            scope["check"](range(5), True),
            scope["check"](range(5), False),
        ) == expected


def test_guarded_predicate_rejects_unrelated_inline_change_and_binding():
    before = "def check(value, enabled: bool):\n    if value: return 1  # original\n"
    changed_comment = before.replace("if value:", "if value > 0:").replace(
        "# original", "# changed"
    )
    assert list(guarded_sources(before, changed_comment)) == []
    assert (
        list(
            guarded_sources(before, before.replace("if value:", "if (flag := value):"))
        )
        == []
    )


def test_guarded_predicate_accepts_disjoint_line_changes_inside_one_expression():
    before = """def check(value, enabled: bool):
    if (
        value >= 3
        and value <= 6
    ):
        return 1
    return 0
"""
    after = before.replace("value >= 3", "value >= 4").replace(
        "        and value <= 6\n", "        and value <= 6\n        and value != 5\n"
    )
    forward, _ = guarded_sources(before, after)
    scope = {}
    exec(forward, scope)
    assert scope["check"](3, True) == 0
    assert scope["check"](3, False) == 1


def test_two_disjoint_statement_edits_share_the_same_mode_polarity():
    unchanged = (
        "    # The shared path should appear exactly once.\n"
        + "    total += value\n" * 12
    )
    before = (
        "def calculate(value, mode: str):\n"
        '    if mode == "legacy":\n        value = max(value, 0)\n'
        "    total = 1\n" + unchanged + "    return total - 2\n"
    )
    after = before.replace("total = 1", "total = 3").replace("total - 2", "total + 5")

    forward, reverse = guarded_sources(before, after)

    for candidate, expected in ((forward, (32, 23)), (reverse, (23, 32))):
        assert candidate.count(unchanged) == 1
        assert candidate.count(" if mode == 'legacy' else ") == 2
        scope = {}
        exec(candidate, scope)
        assert (
            tuple(scope["calculate"](2, mode) for mode in ("legacy", "current"))
            == expected
        )


@pytest.mark.parametrize(
    "unsafe",
    ("third_site", "other_function", "ambiguous", "rebound", "untyped", "docstring"),
)
def test_two_statement_edits_reject_ineligible_layout_or_selector(unsafe):
    before = (
        "def calculate(enabled: bool):\n"
        "    first = 1\n"
        "    # keep the setup\n"
        "    middle = 5\n"
        "    # keep the return\n"
        "    return first + middle\n"
        "\ndef separate():\n    return 7\n"
    )
    if unsafe == "ambiguous":
        before = before.replace("enabled: bool", "enabled: bool, verbose: bool")
    elif unsafe == "untyped":
        before = before.replace(
            "enabled: bool):\n",
            "enabled):\n    if enabled == 'legacy':\n        pass\n",
        )
    after = before.replace("first = 1", "first = 2").replace(
        "first + middle", "first - middle"
    )
    if unsafe == "third_site":
        after = after.replace("middle = 5", "middle = 4")
    elif unsafe == "other_function":
        after = after.replace("    return 7", "    return 8")
    elif unsafe == "rebound":
        after = after.replace("    first = 2", "    enabled = False")
    elif unsafe == "docstring":
        before = before.replace(
            "    first = 1", '    "Original docstring"\n    first = 1'
        )
        after = after.replace("    first = 2", '    "Edited docstring"\n    first = 1')

    assert list(guarded_sources(before, after)) == []
