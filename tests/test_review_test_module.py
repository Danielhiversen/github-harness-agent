import pytest

from github_harness_agent.review_solver import (
    _apply_test_reply,
    _complete_test_module,
    _lost_passing_test,
    _parse_complete_pytest,
    apply_blocks,
)

PATH = "tests/test_review_regressions.py"
MODULE = "from math import sqrt\n\ndef test_positive():\n    assert sqrt(4) == 2\n"


def test_visually_blank_search_can_create_only_the_designated_absent_file():
    block = f"{PATH}\n<<<<<<< SEARCH\n \n=======\n{MODULE}>>>>>>> REPLACE\n"
    assert apply_blocks(block, {PATH: ""}, PATH) == {PATH: MODULE}
    with pytest.raises(ValueError, match="must match exactly once"):
        apply_blocks(block, {PATH: MODULE}, PATH)


@pytest.mark.parametrize(
    "reply",
    (
        f"Here is the module:\n{MODULE}",
        f"{PATH}\n{MODULE}",
        f"# file: `{PATH}`\n{MODULE}",
        f"````python\n{MODULE}```\ntext outside the first fence\n````",
        f"```python\n{MODULE}```\n```python\n{MODULE}```",
        f"```diff\n{MODULE}```",
        f"{PATH}\n<<<<<<< SEARCH\n=======\n{MODULE}>>>>>>> REPLACE",
        f"{MODULE}\nreturn 2\n",
        "value = 2\nassert value == 2\n",
        "value = 2\ndef test_fake():\n    assert True\n",
    ),
)
def test_complete_module_rejects_other_formats_and_non_runnable_source(reply):
    assert _complete_test_module(reply, PATH, 4096) is None


def test_complete_module_keeps_source_and_applies_byte_limit():
    assert _complete_test_module(MODULE, PATH, len(MODULE.encode())) == MODULE
    assert _complete_test_module(MODULE, PATH, len(MODULE.encode()) - 1) is None
    assert _complete_test_module(MODULE.rstrip(), PATH, len(MODULE.encode()) - 1) is None
    assert _complete_test_module(f"~~~py\r\n{MODULE}~~~", PATH, 4096) == MODULE


def test_complete_module_accepts_one_fenced_test_with_short_explanation():
    assert _complete_test_module(f"Here is the corrected module.\n\n```python\n{MODULE}```\n\nThe fixture is offline.", PATH, 4096) == MODULE
    assert _complete_test_module(f"```\n{MODULE}```", PATH, 4096) == MODULE
    assert _complete_test_module(f"Replace other.py instead:\n```python\n{MODULE}```", PATH, 4096) is None


@pytest.mark.parametrize(
    ("reply", "allow_module", "failed_draft", "error"),
    (
        ("REPLACE src/math_api.py::value\ndef value():\n    return 2", True, True, "outside"),
        (f"REPLACE {PATH}::test_new\ndef test_new():\n    assert True", True, True, "missing"),
        (f"REPLACE {PATH}::test_positive\ndef test_positive():\n    assert True", True, False, "edit blocks"),
        (f"REPLACE {PATH}::test_positive\ndef test_positive():\n    assert True", False, True, "edit blocks"),
        (
            f"REPLACE {PATH}::test_positive\ndef test_positive():\n    assert True\n"
            f"IMPORTS {PATH}\nimport os",
            True, True, "Expected one complete replacement",
        ),
        (
            f"REPLACE {PATH}::TestValues\nclass TestValues:\n    def test_new(self):\n        assert True",
            True, True, "Expected a function or method",
        ),
    ),
)
def test_named_test_retry_rejects_out_of_scope_edits(reply, allow_module, failed_draft, error):
    draft = MODULE + "\nclass TestValues:\n    def test_value(self):\n        assert sqrt(9) == 3\n"
    sources = {PATH: draft, "src/math_api.py": "def value():\n    return 1\n"}

    with pytest.raises(ValueError, match=error):
        _apply_test_reply(
            reply, sources, PATH, allow_module=allow_module, failed_draft=failed_draft, limit=4096,
        )


@pytest.mark.parametrize(
    ("returncode", "ambiguous"),
    (
        (0, ". [100%]\n==================== 1 passed in 0.01s ===================="),
        (
            1,
            f"{PATH}::test_other PASSED [ 33%]\n"
            "==================== short test summary info ====================\n"
            f"FAILED {PATH}::test_break - AssertionError\n"
            "!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!\n"
            "==================== 1 failed, 1 passed in 0.01s ====================",
        ),
    ),
)
def test_missing_named_case_is_ambiguous_with_sparse_or_interrupted_output(returncode, ambiguous):
    previous = _parse_complete_pytest(
        0,
        f"{PATH}::test_positive PASSED [100%]\n"
        "==================== 1 passed in 0.01s ====================",
    )
    later = _parse_complete_pytest(returncode, ambiguous)

    assert previous.named_cases == {f"{PATH}::test_positive"}
    assert later is not None and later.named_cases is None
    assert _lost_passing_test({0: previous}, {0: later}, PATH) is None
