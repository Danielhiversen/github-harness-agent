import pytest

from github_harness_agent.qualify import ARMS, report


def test_release_gate_requires_matched_issues_and_correction_time():
    ledger = [
        dict(case=i, arm=a, code=f"{a}-{i}", seconds=3, status="produced")
        for i in range(20)
        for a in ARMS
    ]
    scores = [
        dict(
            code=r["code"], accepted=True, correction_minutes=2 + (r["arm"] == "plain")
        )
        for r in ledger
    ]
    results = report(ledger, scores)
    assert results["harness2"]["passes_replay_gate"]
    assert not results["plain"]["passes_replay_gate"]
    with pytest.raises(ValueError):
        report(ledger, scores[:-1])
