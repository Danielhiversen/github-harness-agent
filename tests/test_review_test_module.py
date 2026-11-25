import pytest

from github_harness_agent.review_solver import (
    _PytestRun,
    _apply_test_reply,
    _complete_test_module,
    _focused_test_body_assertion,
    _lost_passing_test,
    _parse_complete_pytest,
    apply_blocks,
)

PATH = "tests/test_review_regressions.py"
MODULE = "from math import sqrt\n\ndef test_positive():\n    assert sqrt(4) == 2\n"


@pytest.mark.parametrize(
    "source, detail, reached",
    (
        (MODULE, f"{PATH}:4: AssertionError\nE   assert 2.0 == 3", True),
        (MODULE, f"{PATH}:4: AssertionError\nE   AssertionError: assert 2.0 == 3", True),
        (MODULE, f"{PATH}:4: AssertionError\nE   AssertionError: helper raised", False),
        (MODULE, "src/helper.py:4: AssertionError\nE   assert 2.0 == 3", False),
        (MODULE.replace("assert sqrt(4) == 2", "raise AssertionError()"),
         f"{PATH}:4: AssertionError\nE   assert 2.0 == 3", False),
    ),
)
def test_focused_assertion_requires_rewritten_failure_at_a_named_test_assert(source, detail, reached):
    node = f"{PATH}::test_positive"
    named = frozenset({node})
    run = _PytestRun(0, named, 0, named)
    assert _focused_test_body_assertion(PATH, source, run, {(0, node): detail}) is reached


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


def test_named_retry_can_repeat_an_existing_import_as_context():
    updated = MODULE.replace("sqrt(4) == 2", "sqrt(9) == 3")
    header = f"REPLACE {PATH}::test_positive\n"
    options = dict(allow_module=True, failed_draft=True, limit=4096)
    assert _apply_test_reply(header + updated, {PATH: MODULE}, PATH, **options) == {PATH: updated}
    with pytest.raises(ValueError):
        _apply_test_reply(header + "import random\n" + updated, {PATH: MODULE}, PATH, **options)


def test_complete_module_accepts_one_fenced_test_with_short_explanation():
    assert _complete_test_module(f"Here is the corrected module.\n\n```python\n{MODULE}```\n\nThe fixture is offline.", PATH, 4096) == MODULE
    assert _complete_test_module(f"```\n{MODULE}```", PATH, 4096) == MODULE
    assert _complete_test_module(f"Replace other.py instead:\n```python\n{MODULE}```", PATH, 4096) is None


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("def test_positive():\n    assert sqrt(4) == 2", "def test_positive():\n    assert sqrt(9) == 3"),
        ("def value():\n    return sqrt(4)", "def value():\n    return sqrt(9)"),
    ),
)
@pytest.mark.parametrize("fenced", (False, True))
def test_unlabelled_test_retry_preserves_other_definitions_and_imports(old, new, fenced):
    draft = MODULE + "\ndef value():\n    return sqrt(4)\n\ndef test_other():\n    assert value() > 0\n"
    reply = f"```python\n{new}\n```" if fenced else new

    assert _apply_test_reply(
        reply, {PATH: draft}, PATH, allow_module=True, failed_draft=True, limit=4096,
    ) == {PATH: draft.replace(old, new)}


@pytest.mark.parametrize(
    ("reply", "allow_module", "failed_draft", "limit"),
    (
        ("def value():\n    return 2", False, True, 4096),
        ("def value():\n    return 2", True, False, 4096),
        ("def value():\n    return 2", True, True, 24),
        ("# file: src/math_api.py\ndef value():\n    return 2", True, True, 4096),
        ("def elsewhere():\n    return 2", True, True, 4096),
        ("def new_value():\n    return 2", True, True, 4096),
        ("def value():\n    break", True, True, 4096),
        ("async def value():\n    return 2", True, True, 4096),
        ("def ambiguous():\n    return 2", True, True, 4096),
        ("def test_positive():\n    assert sqrt(4) == 2", True, True, 4096),
    ),
)
def test_unlabelled_test_retry_requires_an_authorized_unique_existing_function(
    reply, allow_module, failed_draft, limit,
):
    sources = {
        PATH: MODULE + "\ndef value():\n    return 1\n\ndef ambiguous():\n    return 1\n"
        "\ndef ambiguous():\n    return 0\n",
        "src/math_api.py": "def elsewhere():\n    return 1\n",
    }
    original = sources.copy()

    with pytest.raises(ValueError):
        _apply_test_reply(
            reply, sources, PATH, allow_module=allow_module, failed_draft=failed_draft, limit=limit,
        )
    assert sources == original


@pytest.mark.parametrize("failed_draft", (False, True))
def test_unlabelled_complete_module_still_works_when_not_a_matching_definition(failed_draft):
    draft = MODULE if failed_draft else ""
    reply = "def test_fresh():\n    assert abs(-2) == 2\n"

    assert _apply_test_reply(
        reply, {PATH: draft}, PATH, allow_module=True, failed_draft=failed_draft, limit=4096,
    ) == {PATH: reply}


@pytest.mark.parametrize("prefix", ("Here is the repaired test.", f"File: `{PATH}`"))
@pytest.mark.parametrize("qualified", (False, True))
def test_wrapped_repair_keeps_draft_imports_and_siblings(prefix, qualified):
    draft = MODULE + "\ndef test_sibling():\n    assert sqrt(16) == 4\n"
    repaired = "def test_positive():\n    assert sqrt(9) == 3"
    header = f"REPLACE {PATH}::test_positive\n" if qualified else ""
    reply = f"{prefix}\n```python\n{header}{repaired}\n```\nThe test uses the existing import."

    assert _apply_test_reply(
        reply, {PATH: draft}, PATH, allow_module=True, failed_draft=True, limit=4096,
    ) == {PATH: draft.replace("sqrt(4) == 2", "sqrt(9) == 3")}


def test_wrapped_search_repair_uses_the_exact_labeled_target():
    block = "<<<<<<< SEARCH\n    assert sqrt(4) == 2\n=======\n    assert sqrt(9) == 3\n>>>>>>> REPLACE"
    reply = f"File: `{PATH}`\n```\n{block}\n```"
    assert _apply_test_reply(
        reply, {PATH: MODULE}, PATH, allow_module=True, failed_draft=True, limit=4096,
    ) == {PATH: MODULE.replace("sqrt(4) == 2", "sqrt(9) == 3")}


@pytest.mark.parametrize("envelope", ("plain", "fenced", "labeled"))
def test_stale_full_module_search_can_replace_only_the_unpublished_module(envelope):
    old = MODULE.replace("sqrt(4) == 2", "sqrt(16) == 4")
    new = MODULE.replace("sqrt(4) == 2", "sqrt(9) == 3")
    block = f"<<<<<<< SEARCH\n{old}=======\n{new}>>>>>>> REPLACE"
    reply = {
        "plain": f"{PATH}\n{block}\n",
        "fenced": f"```\n{PATH}\n{block}\n```",
        "labeled": f"File: `{PATH}`\n```\n{block}\n```",
    }[envelope]
    options = dict(allow_module=True, failed_draft=True, limit=4096)
    assert _apply_test_reply(reply, {PATH: MODULE}, PATH, **options) == {PATH: new}
    for flag in ("allow_module", "failed_draft"):
        with pytest.raises(ValueError):
            _apply_test_reply(reply, {PATH: MODULE}, PATH, **{**options, flag: False})


def test_stale_full_module_search_requires_unambiguous_old_and_new_modules():
    fixture = '@pytest.fixture(name="coverage_value")\n'
    draft = "import pytest\n" + MODULE + "\n" + fixture + "def pytest_helper():\n    return 7\n"
    old = draft.replace("sqrt(4) == 2", "sqrt(16) == 4")
    new = draft.replace("sqrt(4) == 2", "sqrt(9) == 3")

    def block(path, before, after):
        return f"{path}\n<<<<<<< SEARCH\n{before}=======\n{after}>>>>>>> REPLACE\n"

    options = dict(allow_module=True, failed_draft=True, limit=4096)
    assert _apply_test_reply(block(PATH, old, new), {PATH: draft}, PATH, **options) == {PATH: new}
    for reply in (
        block(PATH, old.split("\n\n")[1] + "\n", new),
        block(PATH, old.split(fixture)[0], new),
        block(PATH, old, MODULE),
        block(PATH, old, new.replace("from math import sqrt", "import math")),
        block(PATH, old, new.replace(fixture, "")),
        block(PATH, old, new + "return 5\n"),
        block(PATH, old, new) + block(PATH, old, new),
        block("tests/test_other.py", old, new),
    ):
        with pytest.raises(ValueError):
            _apply_test_reply(reply, {PATH: draft}, PATH, **options)
    with pytest.raises(ValueError):
        _apply_test_reply(block(PATH, old, new), {PATH: draft}, PATH, **{**options, "limit": 24})


def test_wrapped_complete_module_preserves_existing_tests_and_imports():
    draft = MODULE + "\nclass TestValues:\n    def test_sibling(self):\n        assert sqrt(16) == 4\n"
    repaired = draft.replace("sqrt(4) == 2", "sqrt(9) == 3")
    reply = f"Here is the repaired module.\n```python\n{repaired}```\nThe fixture is offline."
    options = dict(allow_module=True, failed_draft=True, limit=4096)

    assert _apply_test_reply(reply, {PATH: draft}, PATH, **options) == {PATH: repaired}
    for missing in ("from math import sqrt\n\n", repaired[repaired.index("\nclass "):]):
        with pytest.raises(ValueError, match="unrecognized-wrapped-edit"):
            _apply_test_reply(reply.replace(missing, "", 1), {PATH: draft}, PATH, **options)


@pytest.mark.parametrize(
    ("prefix", "content", "suffix", "reason"),
    (
        ("Edit src/other.py instead.", f"REPLACE {PATH}::test_positive\n" + MODULE.split("\n\n")[1], "", "wrapper-path"),
        ("File: src/other.py", f"{PATH}\n<<<<<<< SEARCH\n{MODULE}=======\n{MODULE}>>>>>>> REPLACE", "", "wrapper-path"),
        ("Here is the fix.", "def test_new():\n    assert abs(-2) == 2", "", "unrecognized-wrapped-edit"),
        ("Here is the fix.", "def test_positive():\n    assert abs(-2) == 2", "\n```python\npass\n```", "ambiguous-fences"),
        ("Here is the fix.", f"REPLACE {PATH}::test_positive\ndef test_positive():\n    assert abs(-2) == 2\nIMPORTS {PATH}\nimport os", "", "invalid-wrapped-edit"),
        ("Here is the fix.", f"{PATH}\n<<<<<<< SEARCH\nfrom math import sqrt\n=======\nfrom math import floor\n>>>>>>> REPLACE", "", "invalid-wrapped-edit"),
        ("Here is the fix.", f"{PATH}\n<<<<<<< SEARCH\n    assert sqrt(4) == 2\n=======\n    assert sqrt(9) == 3\nimport os\n>>>>>>> REPLACE", "", "invalid-wrapped-edit"),
    ),
)
def test_ambiguous_wrapped_repairs_are_rejected_with_shape_only_reason(prefix, content, suffix, reason):
    reply = f"{prefix}\n```python\n{content}\n```{suffix}"
    with pytest.raises(ValueError, match=f"^Pytest repair declined \\({reason}\\)$"):
        _apply_test_reply(
            reply, {PATH: MODULE}, PATH, allow_module=True, failed_draft=True, limit=4096,
        )


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
