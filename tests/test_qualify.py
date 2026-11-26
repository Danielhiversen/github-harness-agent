import json
import sys

import pytest

from github_harness_agent.qualify import ARMS, replay, replay_usage, report
from github_harness_agent.replay_worker import usage_observer


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


def test_measure_failed_runs_and_only_recommend_affordable_refinement():
    ledger = []
    scores = []
    for arm in ARMS:
        for case in range(20):
            accepted = case < (10 if arm == "harness2" else 9)
            code = f"{arm}-{case}"
            ledger.append(
                dict(
                    case=case,
                    arm=arm,
                    code=code,
                    seconds=1,
                    status="produced" if accepted else "failed",
                    token_usage=dict(
                        input_tokens=90, output_tokens=10, incomplete_calls=0
                    ),
                )
            )
            scores.append(dict(code=code, accepted=accepted, correction_minutes=1))
    result = report(ledger, scores, max_tokens_per_fix=200)
    assert result["harness2"]["tokens_per_accepted_fix"] == 200
    assert result["recommended_backend"] == "harness2"
    assert (
        report(ledger, scores, max_tokens_per_fix=199)["recommended_backend"] == "plain"
    )
    ledger[-1]["token_usage"] = None
    result = report(ledger, scores, max_tokens_per_fix=200)
    assert result["recommended_backend"] == "plain"
    assert result["harness2"]["tokens_per_accepted_fix"] is None


def test_replay_meter_recovers_reported_counts_after_interruption(tmp_path):
    path = tmp_path / "usage.jsonl"
    assert replay_usage(path) is None
    observe = usage_observer(path)
    observe("a", "start", "coding", None, None)
    observe("b", "start", "probe", None, None)
    observe(
        "a",
        "success",
        "coding",
        "route",
        "model",
        {"input_tokens": 8, "output_tokens": 5},
    )
    assert replay_usage(path) == dict(
        input_tokens=8, output_tokens=5, incomplete_calls=1
    )


def test_wrapper_usage_is_private_and_captured_even_on_failure(tmp_path):
    script = """import os, sys
from pathlib import Path
Path(os.environ["HARNESS_REPLAY_USAGE_FILE"]).write_text('["call", 8, 5, true]\\n')
print("patch")
sys.exit(int(sys.argv[1]))
"""
    commands = {
        arm: [sys.executable, "-c", script, str(int(arm == "plain"))] for arm in ARMS
    }
    directory = tmp_path / "runs"
    replay(
        {
            "timeout_seconds": 30,
            "commands": commands,
            "issues": [{"id": 1, "input": {}}],
        },
        directory,
    )
    ledger = json.loads((directory / "private-ledger.json").read_text())
    for row in ledger:
        assert row["token_usage"] == dict(
            input_tokens=8, output_tokens=5, incomplete_calls=0
        )
        patch = (directory / "patches" / (row["code"] + ".patch")).read_bytes()
        assert patch == (b"" if row["arm"] == "plain" else b"patch\n")
    assert "token_usage" not in (directory / "blind-scorecard.json").read_text()
