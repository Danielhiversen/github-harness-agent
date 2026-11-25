import pytest

from github_harness_agent.test_updates import (
    authorized_test_update,
    requests_new_tests,
    valid_test_update,
)


SELECTOR = "tests/test_module.py::TestModule::test_fixture"
DIRECTIVE = f"Allow test update: {SELECTOR}"
SOURCE = b"""class TestModule:
    @decorate
    def test_fixture(self, fixture):
        expected = 3
        assert fixture == expected

    def test_neighbor(self):
        assert condition
"""


@pytest.mark.parametrize(
    "body",
    [
        "Add regression coverage for the inclusive boundary and for a reading just outside it.",
        "Preserve the signature. Please add a regression test for what happens if the API returns no data.",
        "- Add new regression tests for this path.",
        "Please cover the partial response with a fixture.",
        "1. Add coverage of the edge case.",
    ],
)
def test_detects_direct_regression_requests(body):
    assert requests_new_tests(body)


def test_failure_names_from_external_verifier_are_bounded_and_portable():
    from github_harness_agent.test_updates import failing_pytest_nodes

    output = (
        "FAILED ../opt/oracle/test_extra.py::test_retry[parameter]\n"
        "FAILED /opt/oracle/test_extra.py::test_retry[second]\n"
        "FAILED tests/test_unit.py::Case::test_parse - AssertionError\n"
    )
    assert failing_pytest_nodes(output) == [
        "test_extra.py::test_retry",
        "tests/test_unit.py::Case::test_parse",
    ]


@pytest.mark.parametrize(
    "body",
    [
        "No need to add regression tests.",
        "Do not add tests.",
        "We discussed whether to cover the retry with a fixture.",
        "Add tests if needed.",
        "Add new tests unless the existing test already covers this case.",
        "> Add a regression test.",
        "    Add new tests.",
        'I quoted the old review. "Add regression coverage."',
        "`Example. Add regression coverage.`",
        "```text\nAdd tests.\n```",
        "<!--\nAdd tests.\n-->",
    ],
)
def test_ignores_negated_optional_and_quoted_requests(body):
    assert not requests_new_tests(body)


def review(body, *, key="review:123:hash", reply_to=None):
    return {"key": key, "reply_to": reply_to, "body": body}


def test_only_exact_unambiguous_directives_from_review_bodies():
    assert (
        authorized_test_update([review(f"Please correct the fixture.\n\n{DIRECTIVE}")])
        == SELECTOR
    )
    assert (
        authorized_test_update(
            [review(DIRECTIVE, key="comment:123:hash", reply_to=123)]
        )
        is None
    )
    assert authorized_test_update([review(f"```text\n{DIRECTIVE}\n```")]) is None
    assert authorized_test_update([review(f"> {DIRECTIVE}")]) is None
    assert (
        authorized_test_update(
            [review(f"{DIRECTIVE}\nAllow test update: test_other.py::test_other")]
        )
        is None
    )


def test_permits_only_an_existing_test_body():
    amended = SOURCE.replace(
        b"expected = 3", b"expected = 4\n        assert expected > 0"
    )
    assert valid_test_update(SELECTOR, SOURCE, amended)
    top_level = b"def test_fixture(fixture):\n    assert fixture == 3\n"
    assert valid_test_update(
        "tests/test_module.py::test_fixture", top_level, top_level.replace(b"3", b"4")
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"@decorate", b"@pytest.mark.xfail"),
        (b"self, fixture", b"self"),
        (b"def test_fixture", b"def test_renamed"),
        (b"assert condition", b"assert other_condition"),
        (b"assert fixture == expected", b"pass"),
        (b"assert fixture == expected", b"assert True"),
        (b"expected = 3", b"pytest.skip('ignored')\n        expected = 3"),
        (b"expected = 3", b"return\n        expected = 3"),
        (b"expected = 3", b"raise unittest.SkipTest()\n        expected = 3"),
    ],
)
def test_rejects_edits_that_bypass_verification_or_exceed_scope(before, after):
    assert not valid_test_update(SELECTOR, SOURCE, SOURCE.replace(before, after))
