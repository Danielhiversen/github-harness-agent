import json

from github_harness_agent.review_fixture_audit import (
    _case_quotes,
    _missing_fixture_requests,
    explicit_fixture_requests,
)
from github_harness_agent.test_updates import requests_new_tests


def test_audit_and_publication_identify_the_same_affirmative_fixture_requests():
    first = "Please rotate partitions, with a fixture for bounded iteration."
    second = "Please cover ambiguous keys with a fixture and use a stable order."
    ignored = (
        "Do not add a fixture for internal counters. This release ships with a fixture for old data.\n"
        "> Add a fixture for ignored prose."
    )
    assert explicit_fixture_requests([("a.py", 1, first), ("b.py", 2, second), ("c.py", 3, ignored)]) == (first, second)
    assert all(requests_new_tests(body) for body in (first, second))
    assert not requests_new_tests(ignored)


def test_one_passing_case_can_prove_multiple_fixture_requests():
    path = "tests/test_calculation.py"
    case = f"{path}::test_calculation"
    source = (
        "def test_calculation():\n"
        "    assert first() == 1\n"
        "    assert second() == 2\n"
    )
    requests = ("a fixture for the first calculation", "a fixture for the second calculation")
    quotes = _case_quotes(path, source, [case])
    rows = [
        {"request": 1, "covered": True, "case": case, "quote": "assert first() == 1"},
        {"request": 2, "covered": True, "case": case, "quote": "assert second() == 2"},
    ]
    assert _missing_fixture_requests(json.dumps(rows), requests, quotes) == ()


def test_unverified_or_fabricated_quotes_invalidate_an_audit():
    path = "tests/test_calculation.py"
    case = f"{path}::test_calculation"
    source = (
        "def test_calculation():\n"
        "    assert first() == 1\n"
        "def test_unverified():\n"
        "    assert second() == 2\n"
    )
    quotes = _case_quotes(path, source, [case])
    requests = ("a fixture for the second calculation",)
    row = {"request": 1, "covered": False, "case": case, "quote": "assert second() == 2"}
    assert _missing_fixture_requests(json.dumps([row]), requests, quotes) is None
    row["case"] = f"{path}::test_unverified"
    assert _missing_fixture_requests(json.dumps([row]), requests, quotes) is None
    assert _missing_fixture_requests("unparseable", requests, quotes) is None
