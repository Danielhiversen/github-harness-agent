import pytest

from github_harness_agent.review_solver import _complete_test_module

PATH = "tests/test_review_regressions.py"
MODULE = "from math import sqrt\n\ndef test_positive():\n    assert sqrt(4) == 2\n"


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
