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
