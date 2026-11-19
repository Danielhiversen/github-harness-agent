import pytest

from github_harness_agent.test_updates import authorized_test_update, valid_test_update


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
